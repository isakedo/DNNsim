
#include <core/BitTactical.h>
#include <iomanip>

namespace core {

    /* SCHEDULER */

    void promote_weight(schedule &dense_filter_schedule, weight_index ineffectual, weight_index candidate) {
        auto inef_time = std::get<0>(ineffectual);
        auto inef_lane = std::get<1>(ineffectual);
        auto cand_time = std::get<0>(candidate);
        auto cand_lane = std::get<1>(candidate);
        auto ineffectual_tuple = dense_filter_schedule[inef_time][inef_lane];
        dense_filter_schedule[inef_time][inef_lane] = dense_filter_schedule[cand_time][cand_lane];
        dense_filter_schedule[cand_time][cand_lane] = ineffectual_tuple;
    }

    template <typename T>
    weights_set BitTactical<T>::L_shape_search(const schedule &dense_schedule, weight_index wgt_idx, int max_time) {

        auto time = std::get<0>(wgt_idx);
        auto lane = std::get<1>(wgt_idx);
        auto upper_bound = (lane/N_LANES)*N_LANES;
        weights_set effectual_candidates;
        auto next_time = time + 1;
        if(next_time >= max_time) return effectual_candidates;

        // Front
        for(int d = 1; d <= LOOKAHEAD_H; d++) {
            auto time_d = time + d;
            if(time_d >= max_time) break;
            auto wgt_tuple = dense_schedule[time_d][lane];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_d,lane));
        }

        // Up
        for(int h = 1; h <= LOOKASIDE_D; h++) {
            auto lane_h = lane - h;
            lane_h = (lane_h) < upper_bound ? N_LANES + lane_h : lane_h; // Wrap around
            auto wgt_tuple = dense_schedule[next_time][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(next_time,lane_h));
        }

        return effectual_candidates;
    }

    // Currently only allowed for H=2 and D=5
    template <typename T>
    weights_set BitTactical<T>::T_shape_search(const schedule &dense_schedule, weight_index wgt_idx, int max_time) {

        auto time = std::get<0>(wgt_idx);
        auto lane = std::get<1>(wgt_idx);
        auto upper_bound = (lane/N_LANES)*N_LANES;
        auto lower_bound = ((lane/N_LANES)+1)*N_LANES;
        weights_set effectual_candidates;
        auto next_time = time + 1;
        if(next_time >= max_time) return effectual_candidates;

        // Front
        for(int d = 1; d <= LOOKAHEAD_H; d++) {
            auto time_d = time + d;
            if(time_d >= max_time) break;
            auto wgt_tuple = dense_schedule[time_d][lane];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_d,lane));
        }

        // Up
        for(int h = 1; h <= LOOKAHEAD_H; h++) {
            auto lane_h = lane - h;
            auto time_h = time + h;
            if(time_h >= max_time) break;
            lane_h = (lane_h) < upper_bound ? N_LANES + lane_h : lane_h; // Wrap around
            auto wgt_tuple = dense_schedule[time_h][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h,lane_h));
        }

        // Down
        for(int h = 1; h <= LOOKAHEAD_H; h++) {
            auto lane_h = lane + h;
            auto time_h = time + h;
            if(time_h >= max_time) break;
            lane_h = (lane_h) >= lower_bound ? lane_h - N_LANES : lane_h; // Wrap around
            auto wgt_tuple = dense_schedule[time_h][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h,lane_h));
        }

        //For free
        auto lane_h = lane - LOOKAHEAD_H - 1;
        lane_h = (lane_h) < upper_bound ? N_LANES + lane_h : lane_h; // Wrap around
        auto wgt_tuple = dense_schedule[next_time][lane_h];
        auto wgt_bits = std::get<3>(wgt_tuple);
        if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(next_time,lane_h));

        return effectual_candidates;

    }

    template <typename T>
    void BitTactical<T>::filter_scheduler(schedule &dense_schedule, int time, int row, int max_time) {

        auto search = SEARCH_SHAPE == 'L' ? &BitTactical<T>::L_shape_search : &BitTactical<T>::T_shape_search;
        int overlap = 1;
        while(overlap > 0) {

            std::vector<int> num_candidates ((unsigned)N_ROWS*N_LANES, 0);
            std::vector<int> min_num_candidates;

            // Get ineffectual weights
            int init_lane = row*N_LANES;
            weights_set ineffectual_weights;
            for(int lane = init_lane; lane < init_lane + N_LANES; lane++) {
                auto wgt_tuple = dense_schedule[time][lane];
                auto wgt_bits = std::get<3>(wgt_tuple);
                if(wgt_bits == 0) ineffectual_weights.push_back(std::make_tuple(time,lane));
            }

            // Num of candidates for each ineffectual weight
            std::vector<weights_set> effectual_candidates ((unsigned)N_ROWS*N_LANES, weights_set());
            for(auto wgt_idx : ineffectual_weights) {
                auto lane = std::get<1>(wgt_idx);
                effectual_candidates[lane] = (this->*search)(dense_schedule,wgt_idx,max_time);
                if(!effectual_candidates[lane].empty()) {
                    num_candidates[lane] = effectual_candidates[lane].size();
                    min_num_candidates.push_back(effectual_candidates[lane].size());
                }
            }

            // Promote less flexible candidates first
            overlap = min_num_candidates.empty() ? -1 : *std::min_element(min_num_candidates.begin(),
                    min_num_candidates.end());
            for(auto wgt_idx : ineffectual_weights) {
                auto lane = std::get<1>(wgt_idx);
                if(num_candidates[lane] == overlap) {
                    //Promote weight
                    auto cand_idx = effectual_candidates[lane].front();
                    promote_weight(dense_schedule,wgt_idx,cand_idx);
                    break;
                }
            }

        }
    }

    bool check_zero_line(const time_schedule &window_schedule) {
        for(auto wgt_tuple : window_schedule) {
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) return false;
        }
        return true;
    }

    template <typename T>
    schedule BitTactical<T>::dense_scheduler(const schedule &sparse_schedule, const std::vector<int> &max_time) {

        schedule dense_schedule = sparse_schedule;
        schedule result_schedule = schedule();

        int skip = 0, max_time_index = 0;
        for(int time=0; time < sparse_schedule.size(); time++) {

            if(max_time[max_time_index] == time) max_time_index++;

            //Skip zero lines
            if(skip < LOOKAHEAD_H && check_zero_line(dense_schedule[time])) {
                skip++;
                continue;
            }

            skip = 0;

            // Divide in filters to process faster
            for(int row = 0; row < N_ROWS; row++) {
                filter_scheduler(dense_schedule, time, row, max_time[max_time_index]);
            }

            result_schedule.push_back(dense_schedule[time]);
        }

        return result_schedule;

    }

    template <typename T>
    schedule BitTactical<T>::sparse_scheduler(const cnpy::Array<T> &wgt, int act_channels, std::vector<int> &max_time) {

        const auto &wgt_shape = wgt.getShape();

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;
        int round_wgt_channels = (int)ceil(wgt_channels/(double)N_LANES)*N_LANES;

        int num_filter_sets = (int)ceil(num_filters/(double)N_ROWS);
        int time_per_filter = (int)ceil(round_wgt_channels*Kx*Ky/(double)N_LANES);
        int total_time = num_filter_sets * time_per_filter;

        schedule sparse_schedule = schedule((unsigned)total_time, time_schedule((unsigned)N_ROWS*N_LANES,
                schedule_tuple(-1,-1,-1,0)));

        for(int m = 0; m < num_filters; m++) {

            // Two towers alexnet
            int start_group = 0;
            if(m >= it_per_group)
                start_group = wgt_channels;

            // Fix for MobileNet
            if(wgt_channels == 1 && act_channels != 1)
                start_group = m;

            int time = max_time.empty() ? 0 : *std::max_element(max_time.begin(),max_time.end());
            for (int i = 0; i < Kx; i++) {
                for (int j = 0; j < Ky; j++) {
                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        int index = 0;
                        for(int channel = k; channel < std::min(k + N_LANES,wgt_channels); channel++) {
                            auto wgt_bits = wgt.get(m,channel,i,j);
                            int pos = (m % N_ROWS) * N_LANES + index;
                            sparse_schedule[time][pos] = std::make_tuple(start_group + channel,i,j,wgt_bits);
                            index++;
                            if(index == N_LANES) {
                                time++;
                                index = 0;
                            }
                        }
                        if(index != 0)
                            time++;
                    }
                }
            }

            if((m % N_ROWS) == (N_ROWS - 1) || m == (num_filters - 1))
                max_time.push_back(time);

        }

        return sparse_schedule;
    }

    template <typename T>
    schedule BitTactical<T>::scheduler(const cnpy::Array<T> &wgt, int act_channels) {
        std::vector<int> max_time;
        const auto &sparse_schedule = sparse_scheduler(wgt,act_channels,max_time);
        const auto &dense_schedule = dense_scheduler(sparse_schedule,max_time);
        return dense_schedule;
    }

    template <typename T>
    std::vector<schedule> BitTactical<T>::network_scheduler(const Network<T> &network) {

        std::vector<schedule> network_schedule;
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {

                cnpy::Array<T> act = layer.getActivations();
                cnpy::Array<T> wgt = layer.getWeights();
                if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

                auto stride = layer.getStride();

                if(wgt.getShape()[1] == 3 && stride > 1) {
                    act.reshape_first_layer_act((uint16_t) stride);
                    wgt.reshape_first_layer_wgt((uint16_t) stride);
                }

                const auto &dense_schedule = scheduler(wgt, act.getShape()[1]);
                network_schedule.push_back(dense_schedule);

            } else if(layer.getType() == "InnerProduct") {

                cnpy::Array<T> wgt = layer.getWeights();
                wgt.reshape_to_4D();
                const auto &dense_schedule = scheduler(wgt, layer.getWeights().getShape()[1]);
                network_schedule.push_back(dense_schedule);

            }
        }

        return network_schedule;

    }

    template class BitTactical<uint16_t>;

}