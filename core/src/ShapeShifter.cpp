
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void ShapeShifter<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear,
            uint64_t COLUMNS) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear, COLUMNS);

        auto GROUPS = ceil(this->column_cycles.size() / (double)GROUP_SIZE);
        this->column_cycles = std::vector<uint64_t>(GROUPS, 0);

        ready_compute_cycle = 0;
        previous_index = 0;
        previous_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
        previous_compute_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
        act_mask = (uint16_t)(1u << (_act_prec - 1u));
    }

    template <typename T>
    uint64_t ShapeShifter<T>::getCycles() const {
        return COLUMN_REGISTERS > 0 || this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string ShapeShifter<T>::name() {
        return TCL ? "BitTacticalP" : DIFFY ? "ShapeShifterDiffy" : "ShapeShifter";
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
        return TCL;
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

        auto max_tile_cycles = 0;
        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tiles_data[t].wgt_row.size() / tiles_data[t].lanes;

            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            auto column_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;

            if (TCL) {
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
                    max_act_bit + 1;

            if (max_tile_cycles < column_cycles) max_tile_cycles = column_cycles;

            this->scheduled_pe += tiles_data[t].filters.size();
            this->idle_pe += ROWS - tiles_data[t].filters.size();

        }

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
    void ShapeShifter<T>::process_convolution(const std::vector<core::TileData<T>> &tiles_data) {

        auto max_group_cycles = std::vector<uint64_t>(this->column_cycles.size(), 0);

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

                if (TCL) {
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

                    if (max_group_cycles[group] < cycles) max_group_cycles[group] = cycles;

                    group++;
                    group_count = 0;
                    max_act_bit = 0;
                    min_act_bit = INT_MAX;
                }
            } // Window

            if (group_count < GROUP_SIZE) {
                auto cycles = MINOR_BIT ?
                        min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                if (max_group_cycles[group] < cycles) max_group_cycles[group] = cycles;
            }

            this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
            this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());

        }

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
    void ShapeShifter<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
    }

    template <typename T>
    bool ShapeShifter<T>::ready() {
        if(this->ready_cycle > *this->global_cycle) this->stall_cycles++;
        return this->ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    bool ShapeShifter<T>::flush() {
        if(this->ready_cycle > *this->global_cycle) this->stall_cycles++;
        return this->done_cycle <= *this->global_cycle;
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
