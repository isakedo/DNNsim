
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Loom<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear);
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
        return "_PG" + std::to_string(PRECISION_GRANULARITY) + "_PSB" + std::to_string(PE_SERIAL_BITS) +
               (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string Loom<T>::header() {
        std::string header = "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        header += "Number of activations processing bits per PE: " + std::to_string(PE_SERIAL_BITS) + "\n";
        header +=  MINOR_BIT ? "Trim bits from the bottom\n" : "";
        return header;
    }

    template <typename T>
    bool Loom<T>::schedule() {
        return false;
    }

    template <typename T>
    void Loom<T>::process_linear(const std::vector<TileData<T>> &tiles_data) {

        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;

            if(this->cycles < this->column_cycles[t][this->column_index]) {
                this->column_stall_cycles += this->column_cycles[t][this->column_index] - this->cycles;
                this->cycles = this->column_cycles[t][this->column_index];
            }

            auto max_bit = 0;
            auto min_bit = 16;
            auto window_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;
            for (int lane = 0; lane < tile_data.lanes; ++lane) {

                auto act_bits = std::get<0>(tile_data.act_row.front()[window_idx + lane]);

                bool act_neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~act_mask;
                    act_neg = true;
                }

                const auto &min_max_act_bits = minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if (act_neg) max_act_bit += 1;

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }

            window_cycles = MINOR_BIT ? min_bit > max_bit ? 1 : max_bit - min_bit + 1 : max_bit + 1;

            auto column_cycles = 0;
            if (DYNAMIC_WEIGHTS) {

            } else {
                column_cycles = window_cycles * this->wgt_prec;
            }

            this->column_cycles[t][this->column_index] = this->cycles + column_cycles;
            this->scheduled_pe += tile_data.filters.size();
            this->idle_pe += ROWS - tile_data.filters.size();
            this->cycles++;

        }

        this->column_index = (this->column_index + 1) % this->column_cycles.size();

    }


    template <typename T>
    void Loom<T>::process_convolution(const std::vector<TileData<T>> &tiles_data) {

        for (const auto &tile_data : tiles_data) {

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.wgt_row.size() / tile_data.lanes;

            auto COLUMN_GROUPS = COLUMNS / PRECISION_GRANULARITY;
            auto window_cycles = std::vector<int>(COLUMN_GROUPS, 0);

            auto group = 0;
            auto max_bit = 0;
            auto min_bit = 16;
            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int lane = 0; lane < tile_data.lanes; ++lane) {

                    auto act_bits = std::get<0>(tile_data.act_row.front()[window_idx + lane]);

                    bool act_neg = false;
                    if((act_bits & act_mask) != 0) {
                        act_bits = act_bits & ~act_mask;
                        act_neg = true;
                    }

                    const auto &min_max_act_bits = minMax(act_bits);

                    auto min_act_bit = std::get<0>(min_max_act_bits);
                    auto max_act_bit = std::get<1>(min_max_act_bits);

                    if (act_neg) max_act_bit += 1;

                    if(min_act_bit < min_bit) min_bit = min_act_bit;
                    if(max_act_bit > max_bit) max_bit = max_act_bit;

                }

                if ((w % PRECISION_GRANULARITY) == 0) {
                    window_cycles[group] = MINOR_BIT ? min_bit > max_bit ? 1 : max_bit - min_bit + 1 : max_bit + 1;

                    group++;
                    max_bit = 0;
                    min_bit = 16;
                }
            }

            if (group < COLUMN_GROUPS)
                window_cycles[group] = MINOR_BIT ? min_bit > max_bit ? 1 : max_bit - min_bit + 1 : max_bit + 1;

            if (DYNAMIC_WEIGHTS) {

            } else {
                auto max_cycles = (int)ceil(sys::get_max(window_cycles) / (double)PE_SERIAL_BITS);
                auto min_cycles = (int)ceil(sys::get_min(window_cycles) / (double)PE_SERIAL_BITS);
                auto wgt_cycles = (int)ceil(this->wgt_prec / (double)PE_SERIAL_BITS);

                this->cycles += max_cycles * wgt_cycles;
                this->pe_stall_cycles += (max_cycles * wgt_cycles) - (min_cycles * wgt_cycles);
            }

            this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
            this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());

        }

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

        if (!DYNAMIC_WEIGHTS) {
            return act_width * this->wgt_prec;
        } else {
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
        }

    }


    template class Loom<uint16_t>;

}