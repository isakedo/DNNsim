
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint64_t Laconic<T>::getCycles() const {
        return this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string Laconic<T>::name() {
        return "Laconic";
    }

    template <typename T>
    void Laconic<T>::dataConversion(base::Array<T> &data) {
        data.powers_of_two_representation();
    }

    /* CYCLES */

    template <typename T>
    std::string Laconic<T>::filename() {
        return Architecture<T>::filename() + (BOOTH_ENCODING ? "_booth" : "");
    }

    template <typename T>
    std::string Laconic<T>::header() {
        std::string header = Architecture<T>::header();
        header += BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
        return header;
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
    void Laconic<T>::process_linear(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_tile_cycles = 0;
        for (int t = 0; t < tiles_data->data.size(); ++t) {
            const auto &tile_data = tiles_data->data[t];

            if (!tile_data.valid)
                continue;

            auto max_cycles = 0;
            auto min_cycles = INT_MAX;
            auto window_idx = this->column_index * tile_data.lanes;

            for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                auto act_shift = this->PE_WIDTH * act_blk;
                auto act_mask = ((1u << this->PE_WIDTH) - 1u) << act_shift;

                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    for (int wgt_blk = 0; wgt_blk < this->wgt_blks; ++wgt_blk) {
                        auto wgt_shift = this->PE_WIDTH * wgt_blk;
                        auto wgt_mask = ((1u << this->PE_WIDTH) - 1u) << wgt_shift;

                        for (int lane = 0; lane < tile_data.lanes; ++lane) {

                            auto act_bits = std::get<0>(tile_data.act_row.front()[window_idx + lane]);
                            act_bits = (act_bits & act_mask) >> act_shift;

                            auto wgt_bits = std::get<0>(tile_data.wgt_row[filter_idx + lane]);
                            wgt_bits = (wgt_bits & wgt_mask) >> wgt_shift;

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

                        } // Multiply weights and activations values
                    } // Wgt Spatial Composition
                } // Filter
            } // Act Spatial Composition

            if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;

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
    void Laconic<T>::process_mmul(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto max_tile_cycles = 0;
        for (const auto &tile_data : tiles_data->data) {

            if (!tile_data.valid)
                continue;

            auto max_cycles = 0;
            auto min_cycles = INT_MAX;

            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                    auto act_shift = this->PE_WIDTH * act_blk;
                    auto act_mask = ((1u << this->PE_WIDTH) - 1u) << act_shift;

                    for (int f = 0; f < tile_data.filters.size(); ++f) {
                        auto filter_idx = f * tile_data.lanes;

                        for (int wgt_blk = 0; wgt_blk < this->wgt_blks; ++wgt_blk) {
                            auto wgt_shift = this->PE_WIDTH * wgt_blk;
                            auto wgt_mask = ((1u << this->PE_WIDTH) - 1u) << wgt_shift;

                            for (int lane = 0; lane < tile_data.lanes; ++lane) {

                                auto act_bits = std::get<0>(tile_data.act_row.front()[window_idx + lane]);
                                act_bits = (act_bits & act_mask) >> act_shift;

                                auto wgt_bits = std::get<0>(tile_data.wgt_row[filter_idx + lane]);
                                wgt_bits = (wgt_bits & wgt_mask) >> wgt_shift;

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

                            } // Multiply weights and activations values
                        } // Wgt Spatial Composition
                    } // Filter
                } // Act Spatial Composition
            } // Window

            if (max_tile_cycles < max_cycles) max_tile_cycles = max_cycles;

            auto scheduled_pe = tile_data.windows.size() * this->act_blks * tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->COLUMNS * this->ROWS - scheduled_pe;

        }

        this->done_cycle = *this->global_cycle + max_tile_cycles;
        this->ready_cycle = *this->global_cycle + max_tile_cycles;
        this->cycles += max_tile_cycles;

    }

    template <typename T>
    void Laconic<T>::process_tiles(const std::shared_ptr<TilesData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_mmul(tiles_data);
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