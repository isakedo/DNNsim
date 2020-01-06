
#include <core/WindowFirstOutS.h>

namespace core {

    /* CYCLES */

    template <typename T>
    std::string WindowFirstOutS<T>::name() {
        return "Window First Output Stationary";
    }

    template <typename T>
    bool WindowFirstOutS<T>::next_dataflow_step(std::vector<TileData<T>> &tiles_data) {

        while (this->current_recurrence < this->recurrence) {

            while (this->group < this->groups) {

                while (this->window_set < this->window_sets) {

                    // Fill window buffer
                    if (!this->window_buffer_filled) {

                        auto window_idx = this->window_set * this->N_COLUMNS;
                        for (int c = 0; c < this->N_COLUMNS; ++c) {

                            auto window = window_idx + c;
                            if (window >= (this->out_x * this->out_y))
                                continue;

                            auto x_window = window % this->out_x;
                            auto y_window = window / this->out_y;
                            this->windows.emplace_back(std::make_tuple(x_window, y_window));
                        }

                        this->fill_window_buffer();
                        this->window_buffer_filled = true;
                    }

                    while (this->filter_set < this->filter_sets) {

                        // Filter set
                        if (!this->filter_buffer_filled) {

                            this->filters = std::vector<std::vector<int>>(this->N_TILES, std::vector<int>());

                            for (int t = 0; t < this->N_TILES; ++t) {

                                auto filter_idx = this->filters_per_group * this->group +
                                        (this->filter_set + t) * this->N_ROWS;

                                auto num_filters = this->wgt->getShape()[0];
                                for (int r = 0; r < this->N_ROWS; ++r) {
                                    auto filter = filter_idx + r;
                                    if (filter >= (this->filters_per_group * (this->group + 1)) || filter >= num_filters)
                                        continue;
                                    this->filters[t].push_back(filter);
                                }

                                this->out_buffer_writes += (this->fc || this->lstm) ? this->filters[t].size() :
                                        this->windows.size() * this->filters[t].size();

                            }

                            this->filter_buffer_filled = true;
                        }

                        bool still_work = false;
                        for (int t = 0; t < this->N_TILES; ++t) {

                            tiles_data[t].valid = false;

                            if (this->filters[t].empty()) break;

                            auto start_time = this->max_buffer_time * this->group;
                            while (this->time[t] < this->max_buffer_time) {

                                this->act_buff_reads++;
                                this->wgt_buff_reads++;
                                this->acc_updates += (this->fc || this->lstm) ? this->filters[t].size() :
                                        this->windows.size() * this->filters[t].size();

                                if (this->schedule) {

                                    // Skip lines of zeroes
                                    bool zero_line = this->scheduler.check_zero_line(this->weight_buffer
                                            [this->filter_sets * this->group + this->filter_set + t][this->time[t]]);
                                    if (this->skip[t] < this->scheduler.getLookaheadH() && zero_line) {
                                        this->skip[t]++;
                                        this->time[t]++;
                                        continue;
                                    }
                                    this->skip[t] = 0;

                                }

                                auto num_act_rows = 1;
                                if (this->schedule) num_act_rows += this->scheduler.getLookaheadH();
                                tiles_data[t].act_row =
                                        BufferSet<T>(this->window_buffer.begin() + start_time + this->time[t],
                                        std::min(this->window_buffer.begin() + start_time + this->time[t] + num_act_rows,
                                                 this->window_buffer.end()));
                                tiles_data[t].wgt_row = this->weight_buffer
                                        [this->filter_sets * this->group + this->filter_set + t][this->time[t]];
                                tiles_data[t].windows = this->windows;
                                tiles_data[t].filters = this->filters[t];
                                tiles_data[t].time = this->time[t];
                                tiles_data[t].lanes = this->N_LANES;
                                tiles_data[t].valid = true;

                                still_work = true;
                                this->time[t]++;
                                break;

                            } // Buffer time

                        } // Tile

                        if (still_work) return true;

                        this->time = std::vector<int>(this->N_TILES, 0);
                        this->filter_buffer_filled = false;
                        this->filters.clear();
                        this->filter_set += this->N_TILES;
                    } // Filter set

                    this->filter_set = 0;
                    this->window_buffer_filled = false;
                    this->windows.clear();
                    this->window_set++;
                } // Window set

                this->window_set = 0;
                this->group++;
            } // Groups

            this->group = 0;
            this->current_recurrence++;
        } // Recurrence

        return false;

    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
