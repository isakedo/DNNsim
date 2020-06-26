
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void ShapeShifter<T>::configure_layer(int _act_prec, int _wgt_prec, int _act_blks, int _wgt_blks,
            int _network_width, bool _signed_act, bool _signed_wgt, bool _linear, uint64_t EF_COLUMNS) {
        Architecture<T>::configure_layer(_act_prec, _wgt_prec, _act_blks, _wgt_blks, _network_width, _signed_act,
                _signed_wgt, _linear, EF_COLUMNS);

        auto GROUPS = ceil(this->column_cycles.size() / (double)GROUP_SIZE);
        this->column_cycles = std::vector<uint64_t>(GROUPS, 0);

        ready_compute_cycle = 0;
        previous_index = 0;
        previous_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
        previous_compute_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
        act_mask = (uint16_t)(1u << (this->PE_WIDTH - 1u));
    }

    template <typename T>
    uint64_t ShapeShifter<T>::getCycles() const {
        return COLUMN_REGISTERS > 0 || this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string ShapeShifter<T>::name() {
        return TCL ? "BitTacticalP" : DIFFY ? "ShapeShifterDiffy" : "ShapeShifter";
    }

    /* CYCLES */

    template <typename T>
    std::string ShapeShifter<T>::filename() {
        return Architecture<T>::filename() + "_GS" + std::to_string(GROUP_SIZE) +
               "_CR" + std::to_string(COLUMN_REGISTERS) + (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string ShapeShifter<T>::header() {
        std::string header = Architecture<T>::header();
        header += "Number of columns per group: " + std::to_string(GROUP_SIZE) + "\n";
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
        return TCL;
    }

    template <typename T>
    void ShapeShifter<T>::process_pe(const BufferSet<T> &act_row, const BufferRow<T> &wgt_row, int window_idx,
            int filter_idx, int lanes, int time, int &min_group_bit, int &max_group_bit, int act_blk) {

        auto shift = this->PE_WIDTH * act_blk;
        auto mask = ((1u << this->PE_WIDTH) - 1u) << shift;

        for (int lane = 0; lane < lanes; ++lane) {

            auto time_h = 0;
            auto lane_d = lane;
            if (!wgt_row.empty()) {
                time_h = std::get<1>(wgt_row[filter_idx + lane]) - time;
                lane_d = std::get<2>(wgt_row[filter_idx + lane]);

                if (time_h < 0) continue;
            }

            auto act_bits = std::get<0>(act_row[time_h][window_idx + lane_d]);
            act_bits = (act_bits & mask) >> shift;

            if(this->signed_act && (act_bits & act_mask) != 0) {
                act_bits = act_bits ^ (act_mask - 1u + act_mask);
            }

            const auto &min_max_bits = minMax(act_bits);

            auto min_bit = std::get<0>(min_max_bits);
            auto max_bit = std::get<1>(min_max_bits);

            max_bit += this->signed_act;

            if(min_bit < min_group_bit) min_group_bit = min_bit;
            if(max_bit > max_group_bit) max_group_bit = max_bit;

        }

    }

    template <typename T>
    void ShapeShifter<T>::process_linear(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_tile_cycles = 0;
        for (int t = 0; t < tiles_data->data.size(); ++t) {
            const auto &tile_data = tiles_data->data[t];

            if (!tile_data.valid)
                continue;

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            auto window_idx = this->column_index * tile_data.lanes;

            for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                if (TCL) {
                    for (int f = 0; f < tile_data.filters.size(); ++f) {
                        auto filter_idx = f * tile_data.lanes;

                        process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx, tile_data.lanes,
                                tile_data.time, min_act_bit, max_act_bit, act_blk);

                    } // Filter
                } else {
                    process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1, min_act_bit,
                            max_act_bit, act_blk);
                }

                group_count++;
                if (group_count >= GROUP_SIZE) {
                    auto column_cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                            max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                    if (max_tile_cycles < column_cycles) max_tile_cycles = column_cycles;

                    group++;
                    group_count = 0;
                    max_act_bit = 0;
                    min_act_bit = INT_MAX;
                }

            } // Act Spatial Composition

            if (group_count < GROUP_SIZE) {
                auto column_cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act
                        : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                if (max_tile_cycles < column_cycles) max_tile_cycles = column_cycles;
            }

            auto scheduled_pe = tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->ROWS - scheduled_pe;

        } // Tile

        if (this->cycles < this->compute_cycles[this->column_index])
            this->cycles = this->compute_cycles[this->column_index];

        this->compute_cycles[this->column_index] = this->cycles + max_tile_cycles;
        this->cycles++;

        this->column_cycles[this->column_index] = *this->global_cycle + max_tile_cycles;
        this->column_index = (this->column_index + 1) % this->column_cycles.size();

        auto new_done_cycle = *this->global_cycle + max_tile_cycles;
        if (new_done_cycle > this->done_cycle) this->done_cycle = new_done_cycle;
        this->ready_cycle = this->column_cycles[this->column_index];

    }

    template <typename T>
    void ShapeShifter<T>::process_mmul(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_group_cycles = std::vector<uint64_t>(this->column_cycles.size(), 0);

        for (int t = 0; t < tiles_data->data.size(); ++t) {
            const auto &tile_data = tiles_data->data[t];

            if (!tile_data.valid)
                continue;

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                    if (TCL) {
                        for (int f = 0; f < tile_data.filters.size(); ++f) {
                            auto filter_idx = f * tile_data.lanes;

                            process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx, tile_data.lanes,
                                    tile_data.time, min_act_bit, max_act_bit, act_blk);

                        } // Filter
                    } else {
                        process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1, min_act_bit,
                                max_act_bit, act_blk);
                    }

                    group_count++;
                    if (group_count >= GROUP_SIZE) {
                        auto cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                                max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                        if (max_group_cycles[group] < cycles) max_group_cycles[group] = cycles;

                        group++;
                        group_count = 0;
                        max_act_bit = 0;
                        min_act_bit = INT_MAX;
                    }
                } // Act Spatial Composition
            } // Window

            if (group_count < GROUP_SIZE) {
                auto cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act
                        : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                if (max_group_cycles[group] < cycles) max_group_cycles[group] = cycles;
            }

            auto scheduled_pe = tile_data.windows.size() * this->act_blks * tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->COLUMNS * this->ROWS - scheduled_pe;

        } // Tile

        // Column registers
        if(COLUMN_REGISTERS > 0) {

            for (int g = 0; g < this->column_cycles.size(); ++g) {
                auto start_time = std::max(*this->global_cycle, this->column_cycles[g]);
                this->column_cycles[g] = start_time + max_group_cycles[g];

                auto start_compute_time = std::max(ready_compute_cycle, this->compute_cycles[g]);
                this->compute_cycles[g] = start_compute_time + max_group_cycles[g];
            }

            ready_compute_cycle = previous_compute_cycles[previous_index];
            this->done_cycle = sys::get_max(this->column_cycles);
            this->ready_cycle = previous_cycles[previous_index];

            previous_compute_cycles[previous_index] = sys::get_max(this->compute_cycles);
            previous_cycles[previous_index] = sys::get_max(this->column_cycles);
            previous_index = (previous_index + 1) % previous_cycles.size();

        } else {

            auto slowest_column = sys::get_max(max_group_cycles);
            this->column_cycles = std::vector<uint64_t>(this->column_cycles.size(),
                    *this->global_cycle + slowest_column);

            this->ready_cycle = *this->global_cycle + slowest_column;
            this->done_cycle = *this->global_cycle + slowest_column;
            this->cycles += slowest_column;

        }

    }

    template <typename T>
    void ShapeShifter<T>::process_tiles(const std::shared_ptr<TilesData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_mmul(tiles_data);
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

        if (TCL) {
            if(wgt == 0) return 0;
        }

        if(this->signed_act && (act & act_mask) != 0) {
            act = act ^ (act_mask - 1u + act_mask);
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        max_act_bit += this->signed_act;

        uint8_t act_width;
        if (MINOR_BIT) act_width = min_act_bit > max_act_bit ? 1 + this->signed_act : max_act_bit - min_act_bit + 1u;
        else act_width = max_act_bit + 1u;

        return act_width * this->network_width;
    }

    template class ShapeShifter<uint16_t>;

}
