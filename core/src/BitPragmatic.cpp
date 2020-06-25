
#include <core/BitPragmatic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void BitPragmatic<T>::configure_layer(int _act_prec, int _wgt_prec, int _act_blks, int _wgt_blks,
            int _network_width, bool _signed_act, bool _signed_wgt, bool _linear, uint64_t EF_COLUMNS) {
        Architecture<T>::configure_layer(_act_prec, _wgt_prec, _act_blks, _wgt_blks, _network_width, _signed_act,
                _signed_wgt, _linear, EF_COLUMNS);

        ready_compute_cycle = 0;
        previous_index = 0;
        previous_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
        previous_compute_cycles = std::vector<uint64_t>(COLUMN_REGISTERS, 0);
    }

    template <typename T>
    uint64_t BitPragmatic<T>::getCycles() const {
        return COLUMN_REGISTERS > 0 || this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string BitPragmatic<T>::name() {
        return TCL ? "BitTacticalE" : DIFFY ? "BitPragmaticDiffy" : "BitPragmatic";
    }

    template <typename T>
    void BitPragmatic<T>::dataConversion(base::Array<T> &data) {
        if (!DIFFY) data.powers_of_two_representation();
    }

    /* CYCLES */

    template <typename T>
    std::string BitPragmatic<T>::filename() {
        return Architecture<T>::filename() + "_B" + std::to_string(BITS_FIRST_STAGE) +
               "_CR" + std::to_string(COLUMN_REGISTERS) + (BOOTH_ENCODING ? "_booth" : "");
    }

    template <typename T>
    std::string BitPragmatic<T>::header() {
        std::string header = Architecture<T>::header();
        header += "Number of bits for first stage shifter: " + std::to_string(BITS_FIRST_STAGE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header +=  BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
        return header;
    }

    template <typename T>
    bool BitPragmatic<T>::diffy() {
        return DIFFY;
    }

    template <typename T>
    bool BitPragmatic<T>::schedule() {
        return TCL;
    }

    template <typename T>
    bool check_act_bits(const std::vector<T> &offsets) {
        for (const auto &act_bits : offsets) {
            if (act_bits != 0) return true;
        }
        return false;
    }

    template <typename T>
    uint16_t BitPragmatic<T>::process_pe(const BufferSet<T> &act_row, const BufferRow<T> &wgt_row, int window_idx,
            int filter_idx, int lanes, int time, int act_blk) {

        auto shift = this->PE_WIDTH * act_blk;
        auto mask = ((1u << this->PE_WIDTH) - 1u) << shift;

        std::vector<T> acts;
        for (int lane = 0; lane < lanes; ++lane) {

            auto time_h = 0;
            auto lane_d = lane;
            if (!wgt_row.empty()) {
                time_h = std::get<1>(wgt_row[filter_idx + lane]) - time;
                lane_d = std::get<2>(wgt_row[filter_idx + lane]);

                if (time_h < 0) continue;
            }

            auto act_bits = std::get<0>(act_row[time_h][window_idx + lane_d]);
            if (DIFFY) act_bits = abs((short)act_bits);
            act_bits = (act_bits & mask) >> shift;

            if (BOOTH_ENCODING) act_bits = booth_encoding(act_bits);

            auto it = std::find(acts.begin(), acts.end(), act_bits);
            if (it == acts.end()) acts.push_back(act_bits);

        }

        // Two stages shifting
        uint16_t cycles = 0;
        auto max_offset_first_stage = (uint8_t)((1u << BITS_FIRST_STAGE) - 1);

        bool still_ones = check_act_bits(acts);
        while (still_ones) {

            // Get the offset for the second stage shift
            int two_stage_offset = INT_MAX;
            for (const auto &act_bits : acts) {
                auto min_bit = std::get<0>(minMax(act_bits));
                if (min_bit != 16 && min_bit < two_stage_offset) two_stage_offset = min_bit;
            }
            // Must be one to enter the while loop
            auto max_offset = two_stage_offset + max_offset_first_stage;

            //Update values
            for (auto &act_bits : acts) {
                for (uint8_t b = two_stage_offset; b <= max_offset; ++b) {
                    auto mask = 1u << b;
                    if ((act_bits & mask) != 0) {
                        act_bits &= ~mask;
                        break;
                    }
                }
            }

            still_ones = check_act_bits(acts);
            cycles++;
        }

        return cycles == 0 ? 1 : cycles;

    }


    template <typename T>
    void BitPragmatic<T>::process_linear(const TilesData<T> &tiles_data) {

        auto max_tile_cycles = 0;
        for (int t = 0; t < tiles_data.tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data.tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto column_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;

            for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                if (TCL) {

                    auto max_cycles = 0;
                    auto min_cycles = INT_MAX;
                    for (int f = 0; f < tile_data.filters.size(); ++f) {
                        auto filter_idx = f * tile_data.lanes;

                        auto cycles = process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx,
                                tile_data.lanes, tile_data.time, act_blk);
                        if (cycles > max_cycles) max_cycles = cycles;
                        if (cycles < min_cycles) min_cycles = cycles;

                    } // Filter

                    column_cycles = max_cycles;

                } else {
                    column_cycles = process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes,
                            -1, act_blk);
                }

                if (max_tile_cycles < column_cycles) max_tile_cycles = column_cycles;

            } // Act Spatial Composition

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
    void BitPragmatic<T>::process_mmul(const TilesData<T> &tiles_data) {

        auto max_column_cycles = std::vector<uint64_t>(this->column_cycles.size(), 0);

        for (int t = 0; t < tiles_data.tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data.tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto idx = 0;
            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;

                for (int act_blk = 0; act_blk < this->act_blks; ++act_blk) {
                    if (TCL) {

                        auto max_cycles = 0;
                        auto min_cycles = INT_MAX;
                        for (int f = 0; f < tile_data.filters.size(); ++f) {
                            auto filter_idx = f * tile_data.lanes;

                            auto cycles = process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx,
                                    tile_data.lanes, tile_data.time, act_blk);
                            if (cycles > max_cycles) max_cycles = cycles;
                            if (cycles < min_cycles) min_cycles = cycles;

                        } // Filter

                        if (max_column_cycles[idx] < max_cycles) max_column_cycles[idx] = max_cycles;
                        idx++;

                    } else {
                        auto cycles = process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes,
                                -1, act_blk);
                        if (max_column_cycles[idx] < cycles) max_column_cycles[idx] = cycles;
                        idx++;
                    }
                } // Act Spatial Composition
            } // Window

            auto scheduled_pe = tile_data.windows.size() * this->act_blks * tile_data.filters.size() * this->wgt_blks;
            this->scheduled_pe += scheduled_pe;
            this->idle_pe += this->COLUMNS * this->ROWS - scheduled_pe;

        } // Tile

        // Column registers
        if(COLUMN_REGISTERS > 0) {

            for (int c = 0; c < this->column_cycles.size(); ++c) {
                auto start_time = std::max(*this->global_cycle, this->column_cycles[c]);
                this->column_cycles[c] = start_time + max_column_cycles[c];

                auto start_compute_time = std::max(ready_compute_cycle, this->compute_cycles[c]);
                this->compute_cycles[c] = start_compute_time + max_column_cycles[c];
            }

            ready_compute_cycle = previous_compute_cycles[previous_index];
            this->done_cycle = sys::get_max(this->column_cycles);
            this->ready_cycle = previous_cycles[previous_index];

            previous_compute_cycles[previous_index] = sys::get_max(this->compute_cycles);
            previous_cycles[previous_index] = sys::get_max(this->column_cycles);
            previous_index = (previous_index + 1) % previous_cycles.size();

        } else {

            auto slowest_column = sys::get_max(max_column_cycles);
            this->column_cycles = std::vector<uint64_t>(this->column_cycles.size(),
                    *this->global_cycle + slowest_column);

            this->ready_cycle = *this->global_cycle + slowest_column;
            this->done_cycle = *this->global_cycle + slowest_column;
            this->cycles += slowest_column;

        }

    }

    template <typename T>
    void BitPragmatic<T>::process_tiles(const TilesData<T> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_mmul(tiles_data);
    }

    /* POTENTIALS */

    template <typename T>
    std::string BitPragmatic<T>::filename_pot() {
        return BOOTH_ENCODING ? "_booth" : "";
    }

    template <typename T>
    std::string BitPragmatic<T>::header_pot() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    uint16_t BitPragmatic<T>::computeBits(T act, T wgt) {

        if (TCL) {
            if(wgt == 0) return 0;
        }

        uint16_t act_bits = act;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        return effectualBits(act_bits) * this->network_width;
    }

    template class BitPragmatic<uint16_t>;

}