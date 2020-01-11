
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Loom<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear,
            uint64_t COLUMNS, uint64_t TILES) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear, COLUMNS, TILES);
        act_mask = (uint16_t)(1u << (_act_prec - 1u));
        wgt_mask = (uint16_t)(1u << (_wgt_prec - 1u));
    }

    template <typename T>
    uint64_t Loom<T>::getCycles() const {
        return this->linear ? sys::get_max(this->column_cycles) : this->cycles;
    }

    template <typename T>
    std::string Loom<T>::name() {
        return DYNAMIC_WEIGHTS ? "DynLoom" : "Loom";
    }

    template <typename T>
    void Loom<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string Loom<T>::filename() {
        return "_GS" + std::to_string(GROUP_SIZE) + "_PSB" + std::to_string(PE_SERIAL_BITS) +
               (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string Loom<T>::header() {
        std::string header = "Number of columns/rows per group: " + std::to_string(GROUP_SIZE) + "\n";
        header += "Number of activations processing bits per PE: " + std::to_string(PE_SERIAL_BITS) + "\n";
        header +=  MINOR_BIT ? "Trim bits from the bottom\n" : "";
        return header;
    }

    template <typename T>
    bool Loom<T>::diffy() {
        return false;
    }

    template <typename T>
    bool Loom<T>::schedule() {
        return false;
    }

    template <typename T>
    void process_pe(const BufferRow<T> &row, int idx, int lanes, uint16_t mask, int &min_group_bit,
            int &max_group_bit) {

        for (int lane = 0; lane < lanes; ++lane) {

            auto bits = std::get<0>(row[idx + lane]);

            bool neg = false;
            if((bits & mask) != 0) {
                bits = bits & ~mask;
                neg = true;
            }

            const auto &min_max_bits = minMax(bits);

            auto min_bit = std::get<0>(min_max_bits);
            auto max_bit = std::get<1>(min_max_bits);

            if (neg) max_bit += 1;

            if(min_bit < min_group_bit) min_group_bit = min_bit;
            if(max_bit > max_group_bit) max_group_bit = max_bit;

        }

    }

    template <typename T>
    void Loom<T>::process_linear(const std::vector<TileData<T>> &tiles_data) {

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

        auto max_pe_stall_cycles = 0;
        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;

            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;

            auto column_cycles = 0;
            auto window_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;

            process_pe(tile_data.act_row.front(), window_idx, tile_data.lanes, act_mask, min_act_bit, max_act_bit);

            window_cycles = MINOR_BIT ? min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

            if (DYNAMIC_WEIGHTS) {

                auto ROW_GROUPS = ceil(ROWS / (double)GROUP_SIZE);
                auto filter_cycles = std::vector<int>(ROW_GROUPS, 0);

                auto group = 0;
                auto group_count = 0;
                auto max_wgt_bit = 0;
                auto min_wgt_bit = INT_MAX;
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    process_pe(tile_data.wgt_row, filter_idx, tile_data.lanes, wgt_mask, min_wgt_bit, max_wgt_bit);

                    group_count++;
                    if (group_count >= GROUP_SIZE) {
                        filter_cycles[group] = MINOR_BIT ?
                                min_wgt_bit > max_wgt_bit ? 1 : max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                        group++;
                        group_count = 0;
                        max_wgt_bit = 0;
                        min_wgt_bit = INT_MAX;
                    }
                }

                if (group_count < GROUP_SIZE)
                    filter_cycles[group] = MINOR_BIT ?
                            min_wgt_bit > max_wgt_bit ? 1 : max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                // Calculate cycles
                auto max_cycles = 0;
                auto min_cycles = INT_MAX;
                for (const auto &wgt_cycles : filter_cycles) {
                    auto act_pe_cycles = (int)ceil(window_cycles / (double)PE_SERIAL_BITS);

                    auto cycles = act_pe_cycles * wgt_cycles;
                    if (cycles > max_cycles) max_cycles = cycles;
                    if (cycles < min_cycles) min_cycles = cycles;
                }

                column_cycles += max_cycles;

                auto pe_stall_cycles = max_cycles - min_cycles;
                if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;

            } else {
                auto act_pe_cycles = (int)ceil(window_cycles / (double)PE_SERIAL_BITS);
                column_cycles = act_pe_cycles * this->wgt_prec;
            }

            this->column_cycles[t][this->column_index] = this->cycles + column_cycles;
            this->scheduled_pe += tile_data.filters.size();
            this->idle_pe += ROWS - tile_data.filters.size();
            this->cycles++;

        }

        this->column_index = (this->column_index + 1) % this->column_cycles.front().size();

        this->pe_stall_cycles += max_pe_stall_cycles;

    }


    template <typename T>
    void Loom<T>::process_convolution(const std::vector<TileData<T>> &tiles_data) {

        auto max_pe_stall_cycles = 0;
        for (const auto &tile_data : tiles_data) {

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.act_row.front().size() / tile_data.lanes;

            auto COLUMN_GROUPS = ceil(COLUMNS / (double)GROUP_SIZE);
            auto window_cycles = std::vector<int>(COLUMN_GROUPS, 0);

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;
            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                process_pe(tile_data.act_row.front(), window_idx, tile_data.lanes, act_mask, min_act_bit, max_act_bit);

                group_count++;
                if (group_count >= GROUP_SIZE) {
                    window_cycles[group] = MINOR_BIT ?
                            min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                    group++;
                    group_count = 0;
                    max_act_bit = 0;
                    min_act_bit = INT_MAX;
                }
            }

            if (group_count < GROUP_SIZE)
                window_cycles[group] = MINOR_BIT ?
                        min_act_bit > max_act_bit ? 1 : max_act_bit - min_act_bit + 1 : max_act_bit + 1;

            if (DYNAMIC_WEIGHTS) {

                auto ROW_GROUPS = ceil(ROWS / (double)GROUP_SIZE);
                auto filter_cycles = std::vector<int>(ROW_GROUPS, 0);

                group = 0;
                group_count = 0;
                auto max_wgt_bit = 0;
                auto min_wgt_bit = 16;
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    process_pe(tile_data.wgt_row, filter_idx, tile_data.lanes, wgt_mask, min_wgt_bit, max_wgt_bit);

                    group_count++;
                    if (group_count >= GROUP_SIZE) {
                        filter_cycles[group] = MINOR_BIT ?
                                min_wgt_bit > max_wgt_bit ? 1 : max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                        group++;
                        group_count = 0;
                        max_wgt_bit = 0;
                        min_wgt_bit = INT_MAX;
                    }
                }

                if (group_count < GROUP_SIZE)
                    filter_cycles[group] = MINOR_BIT ?
                            min_wgt_bit > max_wgt_bit ? 1 : max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                // Calculate cycles
                auto max_cycles = 0;
                auto min_cycles = INT_MAX;
                for (const auto &act_cycles : window_cycles) {
                    for (const auto &wgt_cycles : filter_cycles) {
                        auto act_pe_cycles = (int)ceil(act_cycles / (double)PE_SERIAL_BITS);
                        auto cycles = act_pe_cycles * wgt_cycles;
                        if (cycles > max_cycles) max_cycles = cycles;
                        if (cycles < min_cycles) min_cycles = cycles;
                    }
                }

                this->cycles += max_cycles;
                auto pe_stall_cycles = max_cycles - min_cycles;
                if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;

            } else {
                auto max_act_cycles = (int)ceil(sys::get_max(window_cycles) / (double)PE_SERIAL_BITS);
                auto min_act_cycles = (int)ceil(sys::get_min(window_cycles) / (double)PE_SERIAL_BITS);
                this->cycles += max_act_cycles * this->wgt_prec;

                auto pe_stall_cycles = (max_act_cycles - min_act_cycles) * this->wgt_prec;
                if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;
            }

            this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
            this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());

        }

        this->pe_stall_cycles += max_pe_stall_cycles;

    }

    template <typename T>
    void Loom<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
    }

    /* POTENTIALS */

    template <typename T>
    std::string Loom<T>::filename_pot() {
        return MINOR_BIT ? "_minor" : "";
    }

    template <typename T>
    std::string Loom<T>::header_pot() {
        return MINOR_BIT ? "Trim bits from the bottom\n" : "";
    }

    template <typename T>
    uint16_t Loom<T>::computeBits(T act, T wgt) {

        bool neg_act = false;
        if((act & act_mask) != 0) {
            act = act & ~act_mask;
            neg_act = true;
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        if (neg_act) max_act_bit++;

        auto act_width = MINOR_BIT ? min_act_bit > max_act_bit ? 0 : max_act_bit - min_act_bit + 1u :
                max_act_bit + 1u;

        if (DYNAMIC_WEIGHTS) {

            bool neg_wgt = false;
            if((wgt & wgt_mask) != 0) {
                wgt = wgt & ~wgt_mask;
                neg_wgt = true;
            }

            const auto &min_max_wgt_bits = minMax(wgt);
            auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
            auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
            if (neg_wgt) max_wgt_bit++;

            auto wgt_width = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 0 : max_wgt_bit - min_wgt_bit + 1u :
                    max_wgt_bit + 1u;

            return act_width * wgt_width;
        } else {
            return act_width * this->wgt_prec;
        }

    }


    template class Loom<uint16_t>;

}