
#include <core/WindowFirstOutS.h>

namespace core {

    /* CYCLES */

    template <typename T>
    std::string WindowFirstOutS<T>::name() {
        return "Window First Output Stationary";
    }

    template <typename T>
    void WindowFirstOutS<T>::initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, bool _fc, bool _lstm, int _recurrence,
            int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS,
            uint32_t _N_TILES) {

        Dataflow<T>::initialise_layer(_act, _wgt, _schedule, _fc, _lstm, _recurrence, _out_x, _out_y, _stride, _N_LANES,
                _N_COLUMNS, _N_ROWS, _N_TILES);

        window_sets = (uint64_t)ceil(this->out_x * this->out_y / (double)this->N_COLUMNS);

        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        // Generate weight buffer
        filter_sets = (uint64_t)ceil(num_filters / (double)this->N_ROWS);

        auto round_wgt_channels = (int)ceil(wgt_channels / (double)this->N_LANES) * this->N_LANES;
        max_buffer_time = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)this->N_LANES);

        weight_buffer = Buffer<T>(filter_sets, BufferSet<T>(max_buffer_time, BufferRow<T>(this->N_ROWS * this->N_LANES,
                std::make_tuple(0, 0, 0))));

        this->fill_weight_buffer(weight_buffer);

        // BitTactical schedule
        if (_schedule) {
            this->scheduler.schedule(weight_buffer);
        }

    }

    template <typename T>
    void WindowFirstOutS<T>::initialise_batch(int _batch) {
        Dataflow<T>::initialise_batch(_batch);
        windows = std::vector<WindowCoord>();
        current_recurrence = 0;
        window_set = 0;
        filter_set = 0;
        time = 0;
        skip = 0;
        window_buffer_filled = false;
        filter_buffer_filled = false;
    }

    template <typename T>
    bool WindowFirstOutS<T>::next_dataflow_step(std::vector<TileData<T>> &tiles_data) {

        while (current_recurrence < this->recurrence) {

            while (window_set < window_sets) {

                // Fill window buffer
                if (!window_buffer_filled) {

                    auto window_idx = window_set * this->N_COLUMNS;
                    for (int c = 0; c < this->N_COLUMNS; ++c) {

                        auto window = window_idx + c;
                        if (window >= (this->out_x * this->out_y))
                            continue;

                        auto x_window = window % this->out_x;
                        auto y_window = window / this->out_y;
                        windows.emplace_back(std::make_tuple(x_window, y_window));
                    }

                    window_buffer = BufferSet<T>(max_buffer_time, BufferRow<T>(windows.size() * this->N_LANES,
                            std::make_tuple(0.0f, 0, 0)));

                    this->fill_window_buffer(window_buffer, windows);

                    window_buffer_filled = true;
                }

                while (filter_set < filter_sets) {

                    // Filter set
                    if (!filter_buffer_filled) {

                        auto num_filters = this->wgt->getShape()[0];
                        filters = std::vector<std::vector<int>>(this->N_TILES, std::vector<int>(this->N_ROWS, -1));

                        for (int t = 0; t < this->N_TILES; ++t) {

                            auto filter_idx = (filter_set + t) * this->N_ROWS;
                            for (int r = 0; r < this->N_ROWS; ++r) {
                                auto filter = filter_idx + r;
                                if (filter >= num_filters)
                                    continue;
                                filters[t][r] = filter;
                            }
                        }

                        filter_buffer_filled = true;
                    }

                    while (time < max_buffer_time) {

                        if (this->schedule) {

                            // Skip lines of zeroes
                            bool zero_line = this->scheduler.check_zero_line(window_buffer[time]); //TODO fix
                            if (skip < this->scheduler.getLookaheadH() && zero_line) {
                                skip++;
                                time++;
                                continue;
                            }
                            skip = 0;

                        }

                        // Return current row
                        auto num_filters = this->wgt->getShape()[0];
                        for (int t = 0; t < this->N_TILES; ++t) {

                            auto filter_idx = (filter_set + t) * this->N_ROWS;
                            if (filter_idx >= num_filters) {
                                tiles_data[t].valid = false;
                                continue;
                            }

                            auto num_act_rows = this->schedule ? this->scheduler.getLookaheadH() : 1;
                            tiles_data[t].act_row = BufferSet<T>(window_buffer.begin() + time,
                                    window_buffer.begin() + time + num_act_rows);
                            tiles_data[t].wgt_row = weight_buffer[filter_set + t][time];
                            tiles_data[t].windows = windows;
                            tiles_data[t].filters = filters[t];
                            tiles_data[t].time = time;
                            tiles_data[t].num_act_rows = num_act_rows;
                            tiles_data[t].valid = true;
                        }

                        time++;
                        return true;

                    } // Buffer time

                    time = 0;
                    filter_buffer_filled = false;
                    filters.clear();
                    filter_set += this->N_TILES;
                } // Filter set

                filter_set = 0;
                window_buffer_filled = false;
                windows.clear();
                window_set++;
            } // Window set

            current_recurrence++;
        } // Recurrence

        return false;

    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
