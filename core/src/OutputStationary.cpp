
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
        act_address_map = std::vector<std::vector<std::vector<uint32_t>>>(Ny, std::vector<std::vector<uint32_t>>(Nx,
                std::vector<uint32_t>(ceil(act_channels / (double)this->dram->getValuesPerBlock()))));

        // Column third
        for (int y = 0; y < Ny; ++y) {

            // Row second
            for (int x = 0; x < Nx; ++x) {

                // Store channel-first
                for (int k = 0; k < act_channels; k += this->dram->getValuesPerBlock()) {
                    act_address_map[y][x][k/this->dram->getValuesPerBlock()] =
                            this->dram->getStartActAddress() + next_act_address;
                    next_act_address += 0x40; // Align to 64 bits
                }
            }
        }

        this->act_bank_map = std::vector<std::vector<int>>(Ny, std::vector<int>(Nx));

        int bank = 0;
        int bkp_bank = 0;
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                if (y % this->stride == 0 && x == 0)
                    bank = bkp_bank;
                this->act_bank_map[y][x] = bank;
                bank = (bank + 1) % this->gbuffer->getActBanks();
                if (y % this->stride == 0 && x == out_x * this->stride - 1)
                    bkp_bank = bank;
            }
        }

    }

    template <typename T>
    void OutputStationary<T>::fill_weight_buffer() {

        // Data buffer
        weight_buffer = Buffer<T>(filter_sets, BufferSet<T>(max_buffer_time,
                BufferRow<T>(this->EF_ROWS * this->EF_LANES, std::make_tuple(0, 0, 0))));

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

            auto filter_pos = m % this->EF_ROWS;
            if (filter_pos == 0)
                set_wgt++;

            int buffer_time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < wgt_channels; k += this->EF_ROWS) {
                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t) k + this->EF_LANES, wgt_channels); ++ch) {

                            index = depthwise ? filter_pos : index;
                            auto wgt_bits = this->wgt->get(m, ch, x, y);
                            int pos = filter_pos * this->EF_LANES + index;
                            weight_buffer[set_wgt][buffer_time][pos] = std::make_tuple(wgt_bits, buffer_time, index);

                            if (depthwise) continue;

                            index++;
                            if (index == this->EF_LANES) {
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

        // BitTactical schedule
        if (this->arch->schedule()) {
            this->scheduler->schedule(weight_buffer, this->EF_LANES);
        }

        // Addresses buffer
        auto addresses_per_row = (uint64_t)ceil(this->EF_LANES / (double)this->dram->getValuesPerBlock()) * this->EF_ROWS;
        wgt_address_buffer = AddressBuffer(filter_sets, AddressBufferSet(max_buffer_time,
                AddressBufferRow(addresses_per_row)));

        wgt_address_map.first_address = this->dram->getStartWgtAddress() + next_wgt_address;

        // Filter Set third
        for (int m = 0; m < filter_sets; ++m) {

            // Buffer depth second
            int skip_buf = 0;
            for (int y = 0; y < max_buffer_time; ++y) {

                if (this->arch->schedule()) {
                    bool zero_line = this->scheduler->check_zero_line(this->weight_buffer[m][y]);
                    if (skip_buf < this->scheduler->getLookaheadH() && zero_line) {
                        skip_buf++;
                        continue;
                    }
                    skip_buf = 0;
                }

                // Buffer width first
                for (int x = 0; x < addresses_per_row; ++x) {
                    this->wgt_address_buffer[m][y][x] = this->dram->getStartWgtAddress() + next_wgt_address;
                    this->next_wgt_address += 0x40; // Align to 64 bits
                }
            }
        }

        wgt_address_map.last_address = this->dram->getStartWgtAddress() + next_wgt_address - 0x40;

        // Banks buffer
        auto accesses_per_filter = (uint64_t)ceil(this->EF_LANES * this->dram->getDataSize() /
                (double)this->gbuffer->getBankWidth());
        wgt_bank_buffer = BankBuffer(filter_sets, BankBufferSet(max_buffer_time,
                BankBufferRow(accesses_per_filter * this->EF_ROWS)));

        int bank = 0;
        for (int m = 0; m < filter_sets; ++m) {

            for (int r = 0; r < this->EF_ROWS; ++r) {
                for (int f = 0; f < accesses_per_filter; ++f) {

                    int skip_buf = 0;
                    for (int y = 0; y < max_buffer_time; ++y) {

                        if (this->arch->schedule()) {
                            bool zero_line = this->scheduler->check_zero_line(this->weight_buffer[m][y]);
                            if (skip_buf < this->scheduler->getLookaheadH() && zero_line) {
                                skip_buf++;
                                continue;
                            }
                            skip_buf = 0;
                        }

                        this->wgt_bank_buffer[m][y][r * accesses_per_filter + f] = bank;
                    }

                    bank = (bank + 1) % this->gbuffer->getWgtBanks();

                }
            }
        }

    }

    template <typename T>
    void OutputStationary<T>::fill_window_buffer() {

        auto recurrence = std::static_pointer_cast<OutputStationary<T>::NodeOutS>
                (this->on_chip_graph.front())->recurrence;

        if (windows.empty()) {
            throw std::runtime_error("Window indices may not be empty");
        }

        auto num_windows = this->linear ? this->EF_COLUMNS : windows.size();
        window_buffer = BufferSet<T>(max_buffer_time, BufferRow<T>(num_windows * this->EF_LANES,
                std::make_tuple(0.0f, 0, 0)));

        auto addresses_per_window = (uint64_t)ceil(this->EF_LANES / (double)this->dram->getValuesPerBlock());
        window_address_buffer = AddressBufferSet(max_buffer_time, AddressBufferRow(addresses_per_window *
                windows.size(), 0xFFFFFFFF));

        auto accesses_per_window = (uint64_t)ceil(this->EF_LANES * this->dram->getDataSize() /
                (double)this->gbuffer->getBankWidth());
        window_bank_buffer = BankBufferSet(max_buffer_time, BankBufferRow(accesses_per_window * windows.size(), -1));

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
                    for (int k = 0; k < act_channels; k += this->EF_LANES) {

                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t) k + this->EF_LANES, act_channels); ++ch) {

                            auto act_bits = this->lstm ? this->act->get(recurrence, 0, ch) :
                                    this->act->get(0, ch, x_window + x, y_window + y);

                            if (this->arch->diffy() && !this->linear) {
                                auto prev_act_bits = (x_window - this->stride < 0) ? 0 : this->act->get(0,
                                        ch, x_window + x - this->stride, y_window + y);
                                act_bits = (short)act_bits - (short)prev_act_bits;
                            }

                            auto column = this->linear ? next_column : w;
                            int pos = column * this->EF_LANES + index;
                            window_buffer[buffer_time][pos] = std::make_tuple(act_bits, buffer_time, index);

                            int addr_pos = w * addresses_per_window + index / this->dram->getValuesPerBlock();
                            window_address_buffer[buffer_time][addr_pos] = act_address_map[y_window + y][x_window + x]
                                    [ch / this->dram->getValuesPerBlock()];

                            int bank_pos = w * accesses_per_window +
                                    (index * this->dram->getDataSize()) / this->gbuffer->getBankWidth();
                            window_bank_buffer[buffer_time][bank_pos] = act_bank_map[y_window + y][x_window + x];

                            index++;
                            if (index == this->EF_LANES) {
                                buffer_time++;
                                index = 0;
                            }

                        }
                        if (index != 0) {
                            buffer_time++;
                        }
                        if (this->linear)
                            next_column = (next_column + 1) % this->EF_COLUMNS;

                    } // Activations channel
                } // Kernel X
            } // Kernel Y

        } // Windows

        if (this->linear) windows = std::vector<WindowCoord>(this->EF_COLUMNS, std::make_tuple(0, 0));

    }

    template <typename T>
    void OutputStationary<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _linear, bool _lstm, int _stride, uint32_t _EF_COLUMNS,
            uint32_t _EF_ROWS) {

        Control<T>::configure_layer(_act, _wgt, _linear, _lstm, _stride, _EF_COLUMNS, _EF_ROWS);

        window_set_it = 0;
        filter_set_it = 0;
        time = std::vector<int>(this->arch->getNTiles(), 0);
        skip = std::vector<int>(this->arch->getNTiles(), 0);
        window_buffer_filled = false;
        filter_buffer_filled = false;

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];
        auto Nx = this->lstm ? 1 : act_shape[2];
        auto Ny = this->lstm ? 1 : act_shape[3];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        out_x = (Nx - Kx) / this->stride + 1;
        out_y = (Ny - Ky) / this->stride + 1;

        window_sets = (uint64_t)ceil(this->out_x * this->out_y / (double)this->EF_COLUMNS);
        filter_sets = (uint64_t)ceil(num_filters / (double)this->EF_ROWS);

        // Generate weight buffer
        auto round_wgt_channels = (int)ceil(wgt_channels / (double)this->EF_LANES) * this->EF_LANES;
        max_buffer_time = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)this->EF_LANES);

        fill_weight_buffer();
    }

    INITIALISE_DATA_TYPES(OutputStationary);

}
