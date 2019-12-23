
#include <core/WindowFirstOutS.h>

namespace core {

    /* CYCLES */

    template <typename T>
    std::string WindowFirstOutS<T>::name() {
        return "Window First Output Stationary";
    }

    template <typename T>
    void WindowFirstOutS<T>::initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, uint32_t _N_LANES, uint32_t _N_COLUMNS,
            uint32_t _N_ROWS, uint32_t _N_TILES) {

        Dataflow<T>::initialise_layer(_act, _wgt, _schedule, _N_LANES, _N_COLUMNS, _N_ROWS, _N_TILES);

        // Generate weight buffer
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto filter_sets = (uint64_t)ceil(num_filters / (double)this->N_ROWS);

        auto round_wgt_channels = (int)ceil(wgt_channels / (double)this->N_LANES) * this->N_LANES;
        auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)this->N_LANES);

        weight_buffer = Buffer<T>(filter_sets, BufferSet<T>(time_per_filter, BufferRow<T>(this->N_ROWS * this->N_LANES,
                std::make_tuple(0, 0, 0))));

        this->fill_weight_buffer(weight_buffer, num_filters, wgt_channels, Kx, Ky);

        // BitTactical schedule
        if (_schedule) {
            this->scheduler.schedule(weight_buffer);
        }

    }

    template <typename T>
    void WindowFirstOutS<T>::initialise_batch(int _batch) {
        Dataflow<T>::initialise_batch(_batch);
        x_windows = std::vector<int>();
        y_windows = std::vector<int>();
        x_counter = 0;
        y_counter = 0;
        skip = 0;
    }

    template <typename T>
    bool WindowFirstOutS<T>::next_dataflow_step(BufferRow<T> &act_row, BufferRow<T> &wgt_row,
            std::vector<int> &_x_windows, std::vector<int> &_y_windows, int &_set) {

        /*while(iterateWindows(Ox, Oy, x_windows, y_windows, x_counter, y_counter, this->N_COLUMNS)) {

            // Generate activation window buffer
            auto round_act_channels = (int)ceil(act_channels / (double)this->N_LANES) * this->N_LANES;
            auto time_per_window = (uint64_t)ceil(round_act_channels * Kx * Ky / (double)this->N_LANES);

            auto window_buffer = BufferSet<ValueTuple<T>>(time_per_window,
                    BufferRow<ValueTuple<T>>(x_windows.size() * this->N_LANES, std::make_tuple(0.0f, 0, 0)));

            fill_window_buffer(window_buffer, this->act, x_windows, y_windows, this->batch, act_channels, Kx, Ky,
                    stride);

            if (this->schedule) {
                // Schedule window buffer
            }

            for (int set = 0; set < filter_sets; ++set) {

                // Select window set
                const auto &weight_set = weight_buffer[set];

                for (int time = 0; time < time_per_window; ++time) {

                    if (arch->schedule()) {

                        // Skip lines of zeroes
                        bool zero_line = scheduler.check_zero_line(window_buffer[time]);
                        if (skip < scheduler.getLookaheadH() && zero_line) {
                            skip++;
                            continue;
                        }
                        skip = 0;

                    }

                    const auto &act_row = window_buffer[time];
                    const auto &wgt_row = weight_set[time];
                    // Process tile

                    if (this->CHECK) calculate_output(sim_output, act_row, wgt_row, x_windows,
                                                      y_windows, num_filters, set);

                } // Time of the buffers
            } // Filter sets
        } // Window sets */
        return false;

    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
