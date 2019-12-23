
#include <core/Dataflow.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void Dataflow<T>::fill_weight_buffer(Buffer<T> &weight_buffer, uint64_t num_filters, uint64_t wgt_channels,
            uint64_t Kx, uint64_t Ky) {

        int set_wgt = -1;
        for(int m = 0; m < num_filters; ++m) {

            if ((m % N_ROWS) == 0)
                set_wgt++;

            int time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        int index = 0;
                        for(int ch = k; ch < std::min((uint64_t)k + N_LANES, wgt_channels); ++ch) {

                            auto wgt_bits = wgt->get(m, ch, x, y);
                            int pos = (m % N_ROWS) * N_LANES + index;
                            weight_buffer[set_wgt][time][pos] = std::make_tuple(wgt_bits, time, index);

                            index++;
                            if(index == N_LANES) {
                                time++;
                                index = 0;
                            }
                        } // Channels
                        if(index != 0)
                            time++;
                    } // Channel sets
                } // Kernel Width
            } // Kernel Height

        } // Filter sets

    }

    template <typename T>
    void Dataflow<T>::fill_window_buffer(BufferSet<T> &window_buffer, const std::vector<int> &x_windows,
            const std::vector<int> &y_windows, uint64_t n, uint64_t act_channels, uint64_t Kx, uint64_t Ky,
            int stride) {

        for (int w = 0; w < x_windows.size(); ++w) {
            auto x_window = x_windows[w] * stride;
            auto y_window = y_windows[w] * stride;

            int time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < act_channels; k += N_LANES) {
                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t)k + N_LANES, act_channels); ++ch) {
                            auto act_bits = act->get(n, ch, x_window + x, y_window + y);
                            int pos = w * N_LANES + index;
                            window_buffer[time][pos] = std::make_tuple(act_bits, time, index);
                            index++;
                            if(index == N_LANES) {
                                time++;
                                index = 0;
                            }
                        }
                        if (index != 0) {
                            time++;
                        }
                    } // Activations channel
                } // Kernel X
            } // Kernel Y

        } // Windows

    }

    template <typename T>
    void Dataflow<T>::initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, uint32_t _N_LANES, uint32_t _N_COLUMNS,
            uint32_t _N_ROWS, uint32_t _N_TILES) {
        act = _act;
        wgt = _wgt;
        schedule = _schedule;
        N_LANES = _N_LANES;
        N_COLUMNS = _N_COLUMNS;
        N_ROWS = _N_ROWS;
        N_TILES = _N_TILES;
    }

    template <typename T>
    void Dataflow<T>::initialise_batch(int _batch) {
        batch = _batch;
    }

    INITIALISE_DATA_TYPES(Dataflow);

}
