
#include <core/Dataflow.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void Dataflow<T>::fill_weight_buffer(Buffer<T> &weight_buffer) {

        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

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
    void Dataflow<T>::fill_window_buffer(BufferSet<T> &window_buffer, const std::vector<WindowCoord> &windows) {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = lstm ? act_shape[2] : act_shape[1];

        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        for (int w = 0; w < windows.size(); ++w) {
            auto x_window = std::get<0>(windows[w]) * stride;
            auto y_window = std::get<1>(windows[w]) * stride;

            int time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < act_channels; k += N_LANES) {
                        int index = 0;
                        for (int ch = k; ch < std::min((uint64_t)k + N_LANES, act_channels); ++ch) {
                            auto act_bits = act->get(this->batch, ch, x_window + x, y_window + y);
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
            const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, bool _lstm, int _recurrence, int _out_x,
            int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES) {
        act = _act;
        wgt = _wgt;
        schedule = _schedule;
        lstm = _lstm;
        recurrence = _recurrence;
        out_x = _out_x;
        out_y = _out_y;
        stride = _stride;
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
