
#include <core/OutputStationary.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void OutputStationary<T>::fill_weight_buffer() {

        weight_buffer = Buffer<T>(filter_sets * this->groups, BufferSet<T>(max_buffer_time,
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
        for (int g = 0; g < groups; ++g) {

            for (int m = 0; m < this->filters_per_group; ++m) {

                auto start_group = this->filters_per_group * g;

                if ((start_group + m) >= num_filters)
                    continue;

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
                                auto wgt_bits = this->wgt->get(start_group + m, ch, x, y);
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
        } // Groups

    }

    template <typename T>
    void OutputStationary<T>::fill_window_buffer() {

        if (windows.empty()) {
            throw std::runtime_error("Window indices may not be empty");
        }

        auto num_windows = (this->fc || this->lstm) ? this->N_COLUMNS : windows.size();
        window_buffer = BufferSet<T>(max_window_buffer_time, BufferRow<T>(num_windows * this->N_LANES,
                std::make_tuple(0.0f, 0, 0)));

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        bool depthwise = wgt_channels == 1 && act_channels != 1;
        auto channels = depthwise ? filters_per_group : wgt_channels;

        int next_column = 0;
        for (int w = 0; w < windows.size(); ++w) {
            auto x_window = std::get<0>(windows[w]) * this->stride;
            auto y_window = std::get<1>(windows[w]) * this->stride;

            int buffer_time = 0;
            for (int g = 0; g < groups; ++g) {

                auto start_group = g * channels;

                for (int y = 0; y < Ky; ++y) {
                    for (int x = 0; x < Kx; ++x) {
                        for (int k = 0; k < channels; k += this->N_LANES) {
                            int index = 0;
                            for (int ch = k; ch < std::min((uint64_t) k + this->N_LANES, channels); ++ch) {

                                if ((start_group + ch) >= act_channels)
                                    continue;

                                auto act_bits = this->lstm ? this->act->get(current_recurrence, this->batch, ch) :
                                        this->act->get(this->batch, start_group + ch, x_window + x, y_window + y);

                                if (this->diffy && !this->fc && !this->lstm) {
                                    auto prev_act_bits = (x_window - this->stride < 0) ? 0 : this->act->get(this->batch,
                                            start_group + ch, x_window + x - this->stride, y_window + y);
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
            } // Two towers AlexNet

        } // Windows

        if (this->fc || this->lstm) windows = std::vector<WindowCoord>(this->N_COLUMNS, std::make_tuple(0, 0));

    }

    template <typename T>
    void OutputStationary<T>::initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _diffy, bool _schedule, bool _fc, bool _lstm,
            int _recurrence, int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS,
            uint32_t _N_ROWS, uint32_t _N_TILES) {

        Dataflow<T>::initialise_layer(_act, _wgt, _diffy, _schedule, _fc, _lstm, _recurrence, _out_x, _out_y, _stride,
                _N_LANES, _N_COLUMNS, _N_ROWS, _N_TILES);

        window_sets = (uint64_t)ceil(this->out_x * this->out_y / (double)this->N_COLUMNS);

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        bool depthwise = wgt_channels == 1 && act_channels != 1;

        if (depthwise) {
            auto MIN_DIM = std::min(this->N_LANES, this->N_ROWS);
            this->N_LANES = MIN_DIM;
            this->N_ROWS = MIN_DIM;

            groups = (uint64_t)ceil(num_filters / (double)MIN_DIM);
            filters_per_group = MIN_DIM;
        } else {
            groups = act_channels / wgt_channels == 2 ? 2 : 1;
            filters_per_group = (uint64_t)ceil(num_filters / (double)groups);
        }

        // Generate weight buffer
        filter_sets = (uint64_t)ceil(filters_per_group / (double)this->N_ROWS);

        auto round_wgt_channels = (int)ceil(wgt_channels / (double)this->N_LANES) * this->N_LANES;
        max_buffer_time = depthwise ? (uint64_t)ceil(Kx * Ky) :
                (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)this->N_LANES);

        fill_weight_buffer();

        // BitTactical schedule
        if (_schedule) {
            this->scheduler.schedule(weight_buffer, this->N_LANES);
        }

        max_window_buffer_time = max_buffer_time * groups;

    }

    template <typename T>
    void OutputStationary<T>::initialise_batch(int _batch) {
        Dataflow<T>::initialise_batch(_batch);
        windows = std::vector<WindowCoord>();
        current_recurrence = 0;
        window_set = 0;
        filter_set = 0;
        time = std::vector<int>(this->N_TILES, 0);
        skip = std::vector<int>(this->N_TILES, 0);
        window_buffer_filled = false;
        filter_buffer_filled = false;
    }

    INITIALISE_DATA_TYPES(OutputStationary);

}
