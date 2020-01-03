
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint64_t Laconic<T>::getCycles() const {
        return this->linear ? sys::get_max(this->column_cycles) : this->cycles;
    }

    template <typename T>
    std::string Laconic<T>::name() {
        return "Laconic";
    }

    template <typename T>
    void Laconic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string Laconic<T>::filename() {
        return "";
    }

    template <typename T>
    std::string Laconic<T>::header() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    bool Laconic<T>::diffy() {
        return false;
    }

    template <typename T>
    bool Laconic<T>::schedule() {
        return false;
    }

    template <typename T>
    void Laconic<T>::process_linear(const std::vector<TileData<T>> &tiles_data) {

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

            auto ROWS = tiles_data[t].wgt_row.size() / tiles_data[t].lanes;

            auto max_cycles = 0;
            auto min_cycles = INT_MAX;
            auto window_idx = this->column_index * tiles_data[t].lanes;
            for (int f = 0; f < tiles_data[t].filters.size(); ++f) {
                auto filter_idx = f * tiles_data[t].lanes;

                for (int lane = 0; lane < tiles_data[t].lanes; ++lane) {

                    auto act_bits = std::get<0>(tiles_data[t].act_row.front()[window_idx + lane]);
                    auto wgt_bits = std::get<0>(tiles_data[t].wgt_row[filter_idx + lane]);

                    if (BOOTH_ENCODING) {
                        act_bits = booth_encoding(act_bits);
                        wgt_bits = booth_encoding(wgt_bits);
                    }

                    auto act_cycles = effectualBits(act_bits);
                    auto wgt_cycles = effectualBits(wgt_bits);
                    auto cycles = act_cycles * wgt_cycles;

                    if (cycles == 0) cycles = 1;
                    if (cycles > max_cycles) max_cycles = cycles;
                    if (cycles < min_cycles) min_cycles = cycles;

                } // Multiply 16 weights and 16 activations values
            } // Filter

            this->column_cycles[t][this->column_index] = this->cycles + max_cycles;
            this->scheduled_pe += tiles_data[t].filters.size();
            this->idle_pe += ROWS - tiles_data[t].filters.size();
            this->cycles++;

            auto pe_stall_cycles = max_cycles - min_cycles;
            if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;

        }

        this->column_index = (this->column_index + 1) % this->column_cycles.front().size();

        this->pe_stall_cycles += max_pe_stall_cycles;

    }

    template <typename T>
    void Laconic<T>::process_convolution(const std::vector<TileData<T>> &tiles_data) {

        auto max_pe_stall_cycles = 0;
        for (const auto &tile_data : tiles_data) {

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.act_row.front().size() / tile_data.lanes;

            auto max_cycles = 0;
            auto min_cycles = INT_MAX;

            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    for (int lane = 0; lane < tile_data.lanes; ++lane) {

                        auto act_bits = std::get<0>(tile_data.act_row.front()[window_idx + lane]);
                        auto wgt_bits = std::get<0>(tile_data.wgt_row[filter_idx + lane]);

                        if (BOOTH_ENCODING) {
                            act_bits = booth_encoding(act_bits);
                            wgt_bits = booth_encoding(wgt_bits);
                        }

                        auto act_cycles = effectualBits(act_bits);
                        auto wgt_cycles = effectualBits(wgt_bits);
                        auto cycles = act_cycles * wgt_cycles;

                        if (cycles == 0) cycles = 1;
                        if (cycles > max_cycles) max_cycles = cycles;
                        if (cycles < min_cycles) min_cycles = cycles;

                    } // Multiply 16 weights and 16 activations values
                } // Filter
            } // Window

            this->cycles += max_cycles;
            this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
            this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());

            auto pe_stall_cycles = max_cycles - min_cycles;
            if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;

        }

        this->pe_stall_cycles += max_pe_stall_cycles;

    }

    template <typename T>
    void Laconic<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
    }

    /* POTENTIALS */

    template <typename T>
    std::string Laconic<T>::filename_pot() {
        return BOOTH_ENCODING ? "_booth" : "";
    }

    template <typename T>
    std::string Laconic<T>::header_pot() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    uint16_t Laconic<T>::computeBits(T act, T wgt) {
        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        wgt_bits = BOOTH_ENCODING ? booth_encoding(wgt_bits) : wgt_bits;
        return effectualBits(act_bits) * effectualBits(wgt_bits);
    }


    template class Laconic<uint16_t>;

}