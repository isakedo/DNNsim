
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void ShapeShifter<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear);
        act_mask = (uint16_t)(1u << (_act_prec - 1u));
    }

    template <typename T>
    void ShapeShifter<T>::initialise_batch(uint64_t COLUMNS, uint64_t TILES) {
        Architecture<T>::initialise_batch(COLUMNS, TILES);
        previous_cycles = std::vector<std::vector<uint64_t>>(TILES, std::vector<uint64_t>(COLUMN_REGISTERS, 0));
    }

    template <typename T>
    uint64_t ShapeShifter<T>::getCycles() const {
        return sys::get_max(this->column_cycles);
    }

    template <typename T>
    std::string ShapeShifter<T>::name() {
        return TCT ? "BitTacticalP" : DIFFY ? "ShapeShifterDiffy" : "ShapeShifter";
    }

    template <typename T>
    void ShapeShifter<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        if (!DIFFY) data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string ShapeShifter<T>::filename() {
        return "_GS" + std::to_string(GROUP_SIZE) + "_CR" + std::to_string(COLUMN_REGISTERS) +
               (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string ShapeShifter<T>::header() {
        std::string header = "Number of columns per group: " + std::to_string(GROUP_SIZE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header +=  MINOR_BIT ? "Trim bits from the bottom\n" : "";
        return header;
    }

    template <typename T>
    bool ShapeShifter<T>::diffy() {
        return DIFFY;
    }

    template <typename T>
    bool ShapeShifter<T>::schedule() {
        return TCT;
    }

    template <typename T>
    void ShapeShifter<T>::process_pe(const BufferSet<T> &act_row, const BufferRow<T> &wgt_row, int window_idx,
            int filter_idx, int lanes, int time, int &min_group_bit, int &max_group_bit) {

        for (int lane = 0; lane < lanes; ++lane) {

            auto time_h = 0;
            auto lane_d = lane;
            if (!wgt_row.empty()) {
                time_h = std::get<1>(wgt_row[filter_idx + lane]) - time;
                lane_d = std::get<2>(wgt_row[filter_idx + lane]);

                if (time_h < 0) continue;
            }

            auto act_bits = std::get<0>(act_row[time_h][window_idx + lane_d]);
            if (DIFFY) act_bits = sign_magnitude(act_bits, act_mask);

            bool neg = false;
            if((act_bits & act_mask) != 0) {
                act_bits = act_bits & ~act_mask;
                neg = true;
            }

            const auto &min_max_bits = minMax(act_bits);

            auto min_bit = std::get<0>(min_max_bits);
            auto max_bit = std::get<1>(min_max_bits);

            if (neg) max_bit += 1;

            if(min_bit < min_group_bit) min_group_bit = min_bit;
            if(max_bit > max_group_bit) max_group_bit = max_bit;

        }

    }

    template <typename T>
    void ShapeShifter<T>::process_linear(const std::vector<core::TileData<T>> &tiles_data) {

        auto slowest_column = 0;
        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            auto column_end_cycles = this->column_cycles[t][this->column_index];
            if (column_end_cycles > slowest_column) slowest_column = column_end_cycles;

        }

        if (this->cycles < slowest_column) {
            this->column_stall_cycles += slowest_column - this->cycles;
            this->cycles = slowest_column;
        }

        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tiles_data[t].wgt_row.size() / tiles_data[t].lanes;

            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            auto column_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;

            if (TCT) {
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx, tile_data.lanes,
                            tile_data.time, min_act_bit, max_act_bit);

                } // Filter
            } else {
                process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1, min_act_bit,
                        max_act_bit);
            }

            column_cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 :
                    max_act_bit + 1;;

            this->column_cycles[t][this->column_index] = this->cycles + column_cycles;
            this->scheduled_pe += tiles_data[t].filters.size();
            this->idle_pe += ROWS - tiles_data[t].filters.size();
            this->cycles++;

        }

        this->column_index = (this->column_index + 1) % this->column_cycles.front().size();

    }

    template <typename T>
    void ShapeShifter<T>::process_convolution(const std::vector<core::TileData<T>> &tiles_data) {

        auto max_column_stall_cycles = 0;
        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.act_row.front().size() / tile_data.lanes;

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                if (TCT) {
                    for (int f = 0; f < tile_data.filters.size(); ++f) {
                        auto filter_idx = f * tile_data.lanes;

                        process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx, tile_data.lanes,
                                tile_data.time, min_act_bit, max_act_bit);

                    } // Filter
                } else {
                    process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1, min_act_bit,
                            max_act_bit);
                }

                group_count++;
                if (group_count >= GROUP_SIZE) {
                    auto cycles = MINOR_BIT ?
                            min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                    for (int c = 0; c < GROUP_SIZE; ++c) {
                        this->column_cycles[t][group * GROUP_SIZE + c] += cycles;
                    }

                    group++;
                    group_count = 0;
                    max_act_bit = 0;
                    min_act_bit = INT_MAX;
                }
            } // Window

            if (group_count < GROUP_SIZE) {
                auto cycles = MINOR_BIT ?
                        min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                for (int c = 0; c < GROUP_SIZE; ++c) {
                    auto index = group * GROUP_SIZE + c;
                    if (index >= this->column_cycles[t].size()) break;
                    this->column_cycles[t][index] += cycles;
                }
            }

            this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
            this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());

            // Column registers
            if(COLUMN_REGISTERS > 0) {
                auto fastest_column = previous_cycles[t][0] + 1;
                for(auto &cycles : this->column_cycles[t]) {
                    if(cycles <= previous_cycles[t][0]) {
                        if(cycles < fastest_column) fastest_column = cycles;
                        cycles = previous_cycles[t][0] + 1;
                    }
                }

                auto column_stall_cycles = (previous_cycles[t][0] + 1) - fastest_column;
                if (column_stall_cycles > max_column_stall_cycles) max_column_stall_cycles = column_stall_cycles;

                //Update previous ending cycles
                for(int i = 0; i < COLUMN_REGISTERS - 1; ++i) {
                    previous_cycles[t][i] = previous_cycles[t][i + 1];
                }
                previous_cycles[t][COLUMN_REGISTERS - 1] =
                        *std::max_element(this->column_cycles[t].begin(), this->column_cycles[t].end());
            } else {
                auto slowest_column = *std::max_element(this->column_cycles[t].begin(), this->column_cycles[t].end());
                auto fastest_column = *std::min_element(this->column_cycles[t].begin(), this->column_cycles[t].end());
                this->column_cycles[t] = std::vector<uint64_t>(COLUMNS, slowest_column);

                auto column_stall_cycles = slowest_column - fastest_column;
                if (column_stall_cycles > max_column_stall_cycles) max_column_stall_cycles = column_stall_cycles;
            }

        }

        this->column_stall_cycles += max_column_stall_cycles;

    }

    template <typename T>
    void ShapeShifter<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
    }

    /* POTENTIALS */

    template <typename T>
    std::string ShapeShifter<T>::filename_pot() {
        return MINOR_BIT ? "_minor" : "";
    }

    template <typename T>
    std::string ShapeShifter<T>::header_pot() {
        return MINOR_BIT ? "Trim bits from the bottom\n" : "";
    }

    template <typename T>
    uint16_t ShapeShifter<T>::computeBits(T act, T wgt) {

        if (TCT) {
            if(wgt == 0) return 0;
        }

        bool neg = false;
        if((act & act_mask) != 0) {
            act = act & ~act_mask;
            neg = true;
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        if (neg) max_act_bit++;

        uint8_t act_width;
        if (MINOR_BIT) act_width = min_act_bit > max_act_bit ? 0 : max_act_bit - min_act_bit + 1u;
        else act_width = max_act_bit + 1u;

        return act_width * this->network_bits;
    }

    template class ShapeShifter<uint16_t>;

}
