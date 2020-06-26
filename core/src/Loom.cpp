
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Loom<T>::configure_layer(int _act_prec, int _wgt_prec, int _act_blks, int _wgt_blks, int _network_width,
            bool _signed_act, bool _signed_wgt, bool _linear, uint64_t EF_COLUMNS) {
        Architecture<T>::configure_layer(_act_prec, _wgt_prec, _act_blks, _wgt_blks, _network_width, _signed_act,
                _signed_wgt, _linear, EF_COLUMNS);

        act_mask = (uint16_t)(1u << (this->PE_WIDTH - 1u));
        wgt_mask = (uint16_t)(1u << (this->PE_WIDTH - 1u));
    }

    template <typename T>
    uint64_t Loom<T>::getCycles() const {
        return this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string Loom<T>::name() {
        return DYNAMIC_WEIGHTS ? "DynLoom" : "Loom";
    }

    /* CYCLES */

    template <typename T>
    std::string Loom<T>::filename() {
        return Architecture<T>::filename() + "_GS" + std::to_string(GROUP_SIZE) +
               "_PSB" + std::to_string(PE_SERIAL_BITS) + (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string Loom<T>::header() {
        std::string header = Architecture<T>::header();
        header += "Number of columns/rows per group: " + std::to_string(GROUP_SIZE) + "\n";
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
    void Loom<T>::process_pe(const BufferRow<T> &row, int idx, int lanes, uint16_t n_mask, bool signed_data,
            int &min_group_bit, int &max_group_bit, int blk) {

        auto w_shift = this->PE_WIDTH * blk;
        auto w_mask = ((1u << this->PE_WIDTH) - 1u) << w_shift;

        for (int lane = 0; lane < lanes; ++lane) {

            auto bits = std::get<0>(row[idx + lane]);
            bits = (bits & w_mask) >> w_shift;

            if(signed_data && (bits & n_mask) != 0) {
                bits = bits ^ (n_mask - 1u + n_mask);
            }

            const auto &min_max_bits = minMax(bits);

            auto min_bit = std::get<0>(min_max_bits);
            auto max_bit = std::get<1>(min_max_bits);

            if (signed_data) max_bit += 1;

            if(min_bit < min_group_bit) min_group_bit = min_bit;
            if(max_bit > max_group_bit) max_group_bit = max_bit;

        }

    }

    template <typename T>
    void Loom<T>::process_linear(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_tile_cycles = 0;
        for (int t = 0; t < tiles_data->data.size(); ++t) {
            const auto &tile_data = tiles_data->data[t];

            if (!tile_data.valid)
                continue;

            auto COLUMN_GROUPS = ceil(this->act_blks / (double)GROUP_SIZE);
            auto window_cycles = std::vector<int>(COLUMN_GROUPS, 0);

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;
            auto window_idx = this->column_index * tile_data.lanes;

            for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {

                process_pe(tile_data.act_row.front(), window_idx, tile_data.lanes, act_mask, this->signed_act,
                           min_act_bit, max_act_bit, act_blk);

                group_count++;
                if (group_count >= GROUP_SIZE) {
                    window_cycles[group] = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                            max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                    group++;
                    group_count = 0;
                    max_act_bit = 0;
                    min_act_bit = INT_MAX;
                }

            } // Act Spatial Composition

            if (group_count >= GROUP_SIZE)
                window_cycles[group] = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                        max_act_bit - min_act_bit + 1 : max_act_bit + 1;

            if (DYNAMIC_WEIGHTS) {

                auto ROW_GROUPS = ceil(this->ROWS / (double)GROUP_SIZE);
                auto filter_cycles = std::vector<int>(ROW_GROUPS, 0);

                group = 0;
                group_count = 0;
                auto max_wgt_bit = 0;
                auto min_wgt_bit = INT_MAX;
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    for (int wgt_blk = 0; wgt_blk < this->wgt_blks; ++wgt_blk) {

                        process_pe(tile_data.wgt_row, filter_idx, tile_data.lanes, wgt_mask, this->signed_wgt,
                                min_wgt_bit, max_wgt_bit, wgt_blk);

                        group_count++;
                        if (group_count >= GROUP_SIZE) {
                            filter_cycles[group] = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 1 + this->signed_wgt :
                                    max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                            group++;
                            group_count = 0;
                            max_wgt_bit = 0;
                            min_wgt_bit = INT_MAX;
                        }
                    } // Wgt Spatial Composition
                } // Filter

                if (group_count < GROUP_SIZE)
                    filter_cycles[group] = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 1 + this->signed_wgt :
                            max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

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

                if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;

            } else {
                auto max_act_cycles = (int)ceil(sys::get_max(window_cycles) / (double)PE_SERIAL_BITS);
                auto max_cycles = max_act_cycles * this->wgt_prec;

                if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;
            }

            auto scheduled_pe = tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->ROWS - scheduled_pe;

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
    void Loom<T>::process_mmul(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_tile_cycles = 0;
        for (const auto &tile_data : tiles_data->data) {

            if (!tile_data.valid)
                continue;

            auto COLUMN_GROUPS = ceil(this->COLUMNS / (double)GROUP_SIZE);
            auto window_cycles = std::vector<int>(COLUMN_GROUPS, 0);

            auto group = 0;
            auto group_count = 0;
            auto max_act_bit = 0;
            auto min_act_bit = INT_MAX;
            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {

                    process_pe(tile_data.act_row.front(), window_idx, tile_data.lanes, act_mask, this->signed_act,
                            min_act_bit, max_act_bit, act_blk);

                    group_count++;
                    if (group_count >= GROUP_SIZE) {
                        window_cycles[group] = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                                max_act_bit - min_act_bit + 1 : max_act_bit + 1;

                        group++;
                        group_count = 0;
                        max_act_bit = 0;
                        min_act_bit = INT_MAX;
                    }
                } // Act Spatial Composition
            } // Window

            if (group_count < GROUP_SIZE)
                window_cycles[group] = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                        max_act_bit - min_act_bit + 1 : max_act_bit + 1;

            if (DYNAMIC_WEIGHTS) {

                auto ROW_GROUPS = ceil(this->ROWS / (double)GROUP_SIZE);
                auto filter_cycles = std::vector<int>(ROW_GROUPS, 0);

                group = 0;
                group_count = 0;
                auto max_wgt_bit = 0;
                auto min_wgt_bit = 16;
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    for (int wgt_blk = 0; wgt_blk < this->wgt_blks; ++wgt_blk) {

                        process_pe(tile_data.wgt_row, filter_idx, tile_data.lanes, wgt_mask, this->signed_wgt,
                                min_wgt_bit, max_wgt_bit, wgt_blk);

                        group_count++;
                        if (group_count >= GROUP_SIZE) {
                            filter_cycles[group] = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 1 + this->signed_wgt :
                                    max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

                            group++;
                            group_count = 0;
                            max_wgt_bit = 0;
                            min_wgt_bit = INT_MAX;
                        }
                    } // Wgt Spatial Composition
                } // Filter

                if (group_count < GROUP_SIZE)
                    filter_cycles[group] = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 1 + this->signed_wgt :
                            max_wgt_bit - min_wgt_bit + 1 : max_wgt_bit + 1;

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

                if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;

            } else {
                auto max_act_cycles = (int)ceil(sys::get_max(window_cycles) / (double)PE_SERIAL_BITS);
                auto max_cycles = max_act_cycles * this->wgt_prec;

                if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;
            }

            auto scheduled_pe = tile_data.windows.size() * this->act_blks * tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->COLUMNS * this->ROWS - scheduled_pe;

        }

        this->done_cycle = *this->global_cycle + max_tile_cycles;
        this->ready_cycle = *this->global_cycle + max_tile_cycles;
        this->cycles += max_tile_cycles;

    }

    template <typename T>
    void Loom<T>::process_tiles(const std::shared_ptr<TilesData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_mmul(tiles_data);
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

        if(this->signed_act && (act & act_mask) != 0) {
            act = act ^ (act_mask - 1u + act_mask);
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        max_act_bit += this->signed_act;

        auto act_width = MINOR_BIT ? min_act_bit > max_act_bit ? 1 + this->signed_act :
                max_act_bit - min_act_bit + 1u : max_act_bit + 1u;

        if (DYNAMIC_WEIGHTS) {

            if(this->signed_wgt && (wgt & wgt_mask) != 0) {
                wgt = wgt ^ (wgt_mask - 1u + wgt_mask);
            }

            const auto &min_max_wgt_bits = minMax(wgt);
            auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
            auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
            max_wgt_bit += this->signed_wgt;

            auto wgt_width = MINOR_BIT ? min_wgt_bit > max_wgt_bit ? 1 + this->signed_wgt :
                    max_wgt_bit - min_wgt_bit + 1u : max_wgt_bit + 1u;

            return act_width * wgt_width;
        } else {
            return act_width * this->wgt_prec;
        }

    }


    template class Loom<uint16_t>;

}