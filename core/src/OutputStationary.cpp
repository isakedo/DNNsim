
#include <core/OutputStationary.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void OutputStationary<T>::generate_memory_maps() {

        const std::vector<size_t> &act_shape = this->act->getShape();

        uint64_t act_channels, Nx, Ny;
        if (this->lstm) {
            act_channels = act_shape[2];
            Nx = 1;
            Ny = 1;
        } else {
            act_channels = act_shape[1];
            Nx = act_shape[2];
            Ny = act_shape[3];
        }

        // Generate address map
        auto values_block = 8 / this->data_size;

        act_address_map = std::vector<std::vector<std::vector<uint64_t>>>(Ny, std::vector<std::vector<uint64_t>>(Nx,
                std::vector<uint64_t>(ceil(act_channels / (double)values_block))));

        // Column third
        for (int y = 0; y < Ny; ++y) {

            // Row second
            for (int x = 0; x < Nx; ++x) {

                // Store channel-first
                for (int k = 0; k < act_channels; k += values_block) {
                    act_address_map[y][x][k/values_block] = this->start_act_address + next_act_address;
                    next_act_address += 0x40; // Align to 64 bits
                }
            }
        }

        auto values_per_row = (uint64_t)ceil(this->N_ROWS * this->N_LANES / (double)values_block);
        wgt_address_map = std::vector<std::vector<std::vector<uint64_t>>>(filter_sets,
                std::vector<std::vector<uint64_t>>(max_buffer_time, std::vector<uint64_t>(values_per_row)));

        // Filter Set third
        for (int m = 0; m < filter_sets; ++m) {

            // Buffer depth second
            int skip_buf = 0;
            for (int y = 0; y < max_buffer_time; ++y) {

                if (this->schedule) {
                    bool zero_line = this->scheduler.check_zero_line(this->weight_buffer[m][y]);
                    if (skip_buf < this->scheduler.getLookaheadH() && zero_line) {
                        skip_buf++;
                        continue;
                    }
                    skip_buf = 0;
                }

                // Buffer width first
                for (int x = 0; x < values_per_row; ++x) {
                    this->wgt_address_map[m][y][x] = this->start_wgt_address + next_wgt_address;
                    this->next_wgt_address += 0x40; // Align to 64 bits
                }
            }
        }

        auto BANKS = this->global_buffer_banks / 2;
        this->act_bank_map = std::vector<std::vector<int>>(Ny, std::vector<int>(Nx));

        int bank = 0;
        int bkp_bank = 0;
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                if (y % this->stride == 0 && x == 0)
                    bank = bkp_bank;
                this->act_bank_map[y][x] = bank;
                bank = (bank + 1) % BANKS;
                if (y % this->stride == 0 && x == this->out_x * this->stride - 1)
                    bkp_bank = bank;
            }
        }

    }

    template <typename T>
    void OutputStationary<T>::fill_weight_buffer() {

        weight_buffer = Buffer<T>(filter_sets, BufferSet<T>(max_buffer_time,
                BufferRow<T>(this->N_ROWS * this->N_LANES, std::make_tuple(0, 0, 0))));

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        bool depthwise = wgt_channels == 1 && act_channels != 1;

        int set_wgt = -1;
        for (int m = 0; m < num_filters; ++m) {

            auto filter_pos = m % this->N_ROWS;
            if (filter_pos == 0)
                set_wgt++;

            int buffer_time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < wgt_channels; k += this->N_LANES) {
                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t) k + this->N_LANES, wgt_channels); ++ch) {

                            index = depthwise ? filter_pos : index;
                            auto wgt_bits = this->wgt->get(m, ch, x, y);
                            int pos = filter_pos * this->N_LANES + index;
                            weight_buffer[set_wgt][buffer_time][pos] = std::make_tuple(wgt_bits, buffer_time, index);

                            if (depthwise) continue;

                            index++;
                            if (index == this->N_LANES) {
                                buffer_time++;
                                index = 0;
                            }
                        } // Channels
                        if (depthwise || index != 0)
                            buffer_time++;
                    } // Channel sets
                } // Kernel Width
            } // Kernel Height

        } // Filter sets

    }

    template <typename T>
    void OutputStationary<T>::fill_window_buffer() {

        auto recurrence = on_chip_graph.front().recurrence;

        if (windows.empty()) {
            throw std::runtime_error("Window indices may not be empty");
        }

        auto num_windows = (this->fc || this->lstm) ? this->N_COLUMNS : windows.size();
        window_buffer = BufferSet<T>(max_buffer_time, BufferRow<T>(num_windows * this->N_LANES,
                std::make_tuple(0.0f, 0, 0)));

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        int next_column = 0;
        for (int w = 0; w < windows.size(); ++w) {
            auto x_window = std::get<0>(windows[w]) * this->stride;
            auto y_window = std::get<1>(windows[w]) * this->stride;

            int buffer_time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < act_channels; k += this->N_LANES) {
                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t) k + this->N_LANES, act_channels); ++ch) {

                            auto act_bits = this->lstm ? this->act->get(recurrence, 0, ch) :
                                    this->act->get(0, ch, x_window + x, y_window + y);

                            if (this->diffy && !this->fc && !this->lstm) {
                                auto prev_act_bits = (x_window - this->stride < 0) ? 0 : this->act->get(0,
                                        ch, x_window + x - this->stride, y_window + y);
                                act_bits = (short)act_bits - (short)prev_act_bits;
                            }

                            auto column = (this->fc || this->lstm) ? next_column : w;
                            int pos = column * this->N_LANES + index;
                            window_buffer[buffer_time][pos] = std::make_tuple(act_bits, buffer_time, index);

                            index++;
                            if (index == this->N_LANES) {
                                buffer_time++;
                                index = 0;
                            }
                        }
                        if (index != 0) {
                            buffer_time++;
                        }
                        if (this->fc || this->lstm)
                            next_column = (next_column + 1) % this->N_COLUMNS;
                    } // Activations channel
                } // Kernel X
            } // Kernel Y

        } // Windows

        if (this->fc || this->lstm) windows = std::vector<WindowCoord>(this->N_COLUMNS, std::make_tuple(0, 0));

    }

    template <typename T>
    void OutputStationary<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _diffy, bool _schedule, bool _fc, bool _lstm,
            int _recurrence, int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS,
            uint32_t _N_ROWS, uint32_t _N_TILES) {

        Control<T>::configure_layer(_act, _wgt, _diffy, _schedule, _fc, _lstm, _recurrence, _out_x, _out_y, _stride,
                _N_LANES, _N_COLUMNS, _N_ROWS, _N_TILES);

        window_set_it = 0;
        filter_set_it = 0;
        time = std::vector<int>(this->N_TILES, 0);
        skip = std::vector<int>(this->N_TILES, 0);
        window_buffer_filled = false;
        filter_buffer_filled = false;

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        window_sets = (uint64_t)ceil(this->out_x * this->out_y / (double)this->N_COLUMNS);
        filter_sets = (uint64_t)ceil(num_filters / (double)this->N_ROWS);

        // Generate weight buffer
        auto round_wgt_channels = (int)ceil(wgt_channels / (double)this->N_LANES) * this->N_LANES;
        max_buffer_time = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)this->N_LANES);

        fill_weight_buffer();

        // BitTactical schedule
        if (_schedule) {
            this->scheduler.schedule(weight_buffer, this->N_LANES);
        }

    }

    template <typename T>
    bool OutputStationary<T>::still_off_chip_data() {
        on_chip_graph.erase(on_chip_graph.begin());
        return !on_chip_graph.empty();
    }

    INITIALISE_DATA_TYPES(OutputStationary);

}
