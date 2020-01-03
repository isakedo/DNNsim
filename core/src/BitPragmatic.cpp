
#include <core/BitPragmatic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void BitPragmatic<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear);
        act_mask = (uint16_t)(1u << (_act_prec - 1u));
    }

    template <typename T>
    void BitPragmatic<T>::initialise_batch(uint64_t COLUMNS, uint64_t TILES) {
        Architecture<T>::initialise_batch(COLUMNS, TILES);
        previous_cycles = std::vector<std::vector<uint64_t>>(TILES, std::vector<uint64_t>(COLUMN_REGISTERS, 0));
    }

    template <typename T>
    uint64_t BitPragmatic<T>::getCycles() const {
        return sys::get_max(this->column_cycles);
    }

    template <typename T>
    std::string BitPragmatic<T>::name() {
        return TCT ? "BitTacticalE" : DIFFY ? "BitPragmaticDiffy" : "BitPragmatic";
    }

    template <typename T>
    void BitPragmatic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        if (!DIFFY) data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string BitPragmatic<T>::filename() {
        return "_B" + std::to_string(BITS_FIRST_STAGE) + "_CR" + std::to_string(COLUMN_REGISTERS) +
               (BOOTH_ENCODING ? "_booth" : "");
    }

    template <typename T>
    std::string BitPragmatic<T>::header() {
        std::string header = "Number of bits for first stage shifter: " + std::to_string(BITS_FIRST_STAGE) + "\n";
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
        return TCT;
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
            int filter_idx, int lanes, int time) {

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
            if (DIFFY) act_bits = sign_magnitude(act_bits, act_mask);
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
    void BitPragmatic<T>::process_linear(const std::vector<core::TileData<T>> &tiles_data) {

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

            auto column_cycles = 0;
            auto window_idx = this->column_index * tile_data.lanes;

            if (TCT) {

                auto max_cycles = 0;
                auto min_cycles = INT_MAX;
                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;

                    auto cycles = process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx,
                            tile_data.lanes, tile_data.time);
                    if (cycles > max_cycles) max_cycles = cycles;
                    if (cycles < min_cycles) min_cycles = cycles;

                } // Filter

                column_cycles = max_cycles;
                auto pe_stall_cycles = max_cycles - min_cycles;
                if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;


            } else {
               column_cycles = process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1);
            }

            this->column_cycles[t][this->column_index] = this->cycles + column_cycles;
            this->scheduled_pe += tiles_data[t].filters.size();
            this->idle_pe += ROWS - tiles_data[t].filters.size();
            this->cycles++;

        }

        this->column_index = (this->column_index + 1) % this->column_cycles.front().size();

        this->pe_stall_cycles += max_pe_stall_cycles;

    }

    template <typename T>
    void BitPragmatic<T>::process_convolution(const std::vector<core::TileData<T>> &tiles_data) {

        auto max_pe_stall_cycles = 0;
        auto max_column_stall_cycles = 0;
        for (int t = 0; t < tiles_data.size(); ++t) {
            const auto &tile_data = tiles_data[t];

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.act_row.front().size() / tile_data.lanes;

            if (TCT) {

                for (int w = 0; w < tile_data.windows.size(); ++w) {
                    auto window_idx = w * tile_data.lanes;

                    auto max_cycles = 0;
                    auto min_cycles = INT_MAX;
                    for (int f = 0; f < tile_data.filters.size(); ++f) {
                        auto filter_idx = f * tile_data.lanes;

                        auto cycles = process_pe(tile_data.act_row, tile_data.wgt_row, window_idx, filter_idx,
                                tile_data.lanes, tile_data.time);
                        if (cycles > max_cycles) max_cycles = cycles;
                        if (cycles < min_cycles) min_cycles = cycles;

                    } // Filter

                    this->column_cycles[t][w] += max_cycles;

                    auto pe_stall_cycles = max_cycles - min_cycles;
                    if (pe_stall_cycles > max_pe_stall_cycles) max_pe_stall_cycles = pe_stall_cycles;

                } // Window

            } else {
                for (int w = 0; w < tile_data.windows.size(); ++w) {
                    auto window_idx = w * tile_data.lanes;
                    auto cycles = process_pe(tile_data.act_row, BufferRow<T>(), window_idx, -1, tile_data.lanes, -1);
                    this->column_cycles[t][w] += cycles;
                } // Window
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

        this->pe_stall_cycles += max_pe_stall_cycles;
        this->column_stall_cycles += max_column_stall_cycles;

    }

    template <typename T>
    void BitPragmatic<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
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

        if (TCT) {
            if(wgt == 0) return 0;
        }

        uint16_t act_bits = act;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        return effectualBits(act_bits) * this->network_bits;
    }

    template class BitPragmatic<uint16_t>;

}