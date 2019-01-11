
#include <core/BitTactical.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    bool BitTactical<T>::check_schedule(const schedule &dense_schedule, int init_filter, int max_filter) {

        for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
            if(!dense_schedule[filter].empty())
                return true;
        }
        return false;
    }

    template <typename T>
    void BitTactical<T>::update_schedule(schedule &dense_schedule, int init_filter, int max_filter) {

        for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
            if(!dense_schedule[filter].empty()) dense_schedule[filter].erase(dense_schedule[filter].begin());
        }
    }

    /* SCHEDULER */

    void promote_weight(filter_schedule &dense_filter_schedule, weight_index ineffectual, weight_index candidate) {
        auto inef_time = std::get<0>(ineffectual);
        auto inef_lane = std::get<1>(ineffectual);
        auto cand_time = std::get<0>(candidate);
        auto cand_lane = std::get<1>(candidate);
        dense_filter_schedule[inef_time][inef_lane] = dense_filter_schedule[cand_time][cand_lane];
        dense_filter_schedule[cand_time][cand_lane] = std::make_tuple(0,0,0,0);
    }

    weights_set L_shape_search(const filter_schedule &dense_filter_schedule, weight_index wgt_idx, int LOOKAHEAD_H,
            int LOOKASIDE_D) {

        auto time = std::get<0>(wgt_idx);
        auto lane = std::get<1>(wgt_idx);
        weights_set effectual_candidates;
        auto next_time = time + 1;
        if(next_time >= dense_filter_schedule.size()) return effectual_candidates;

        // Front
        for(int d = 1; d <= LOOKAHEAD_H; d++) {
            auto time_d = time + d;
            if(time_d >= dense_filter_schedule.size()) break;
            auto wgt_tuple = dense_filter_schedule[time_d][lane];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_d,lane));
        }

        // Up
        for(int h = 1; h <= LOOKASIDE_D; h++) {
            auto lane_h = lane - h;
            lane_h = (lane_h) < 0 ? WEIGHT_LANES + lane_h : lane_h; // Wrap around
            auto wgt_tuple = dense_filter_schedule[next_time][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(next_time,lane_h));
        }

        return effectual_candidates;
    }

    // Currently only allowed for H=2 and D=5
    weights_set T_shape_search(const filter_schedule &dense_filter_schedule, weight_index wgt_idx, int LOOKAHEAD_H,
            int LOOKASIDE_D) {

        auto time = std::get<0>(wgt_idx);
        auto lane = std::get<1>(wgt_idx);
        weights_set effectual_candidates;
        auto next_time = time + 1;
        if(next_time >= dense_filter_schedule.size()) return effectual_candidates;

        // Front
        for(int d = 1; d <= LOOKAHEAD_H; d++) {
            auto time_d = time + d;
            if(time_d >= dense_filter_schedule.size()) break;
            auto wgt_tuple = dense_filter_schedule[time_d][lane];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_d,lane));
        }

        // Up
        for(int h = 1; h <= LOOKAHEAD_H; h++) {
            auto lane_h = lane - h;
            auto time_h = time + h;
            if(time_h >= dense_filter_schedule.size()) break;
            lane_h = (lane_h) < 0 ? WEIGHT_LANES + lane_h : lane_h; // Wrap around
            auto wgt_tuple = dense_filter_schedule[time_h][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h,lane_h));
        }

        // Down
        for(int h = 1; h <= LOOKAHEAD_H; h++) {
            auto lane_h = lane + h;
            auto time_h = time + h;
            if(time_h >= dense_filter_schedule.size()) break;
            lane_h = (lane_h) >= WEIGHT_LANES ? lane_h - WEIGHT_LANES : lane_h; // Wrap around
            auto wgt_tuple = dense_filter_schedule[time_h][lane_h];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h,lane_h));
        }

        //For free
        auto lane_h = lane - LOOKAHEAD_H - 1;
        lane_h = (lane_h) < 0 ? WEIGHT_LANES + lane_h : lane_h; // Wrap around
        auto wgt_tuple = dense_filter_schedule[next_time][lane_h];
        auto wgt_bits = std::get<3>(wgt_tuple);
        if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(next_time,lane_h));

        return effectual_candidates;

    }

    template <typename T>
    void BitTactical<T>::filter_scheduler(filter_schedule &dense_filter_schedule, int time) {

        auto search = SEARCH_SHAPE == 'L' ? L_shape_search : T_shape_search;
        int overlap = 1;
        while(overlap > 0) {

            std::vector<int> num_candidates (WEIGHT_LANES, 0);
            std::vector<int> min_num_candidates;

            // Get ineffectual weights
            weights_set ineffectual_weights;
            const time_schedule &window_schedule = dense_filter_schedule[time];
            for(int lane = 0; lane < window_schedule.size(); lane++) {
                auto wgt_tuple = window_schedule[lane];
                auto wgt_bits = std::get<3>(wgt_tuple);
                if(wgt_bits == 0) ineffectual_weights.push_back(std::make_tuple(time,lane));
            }

            // Num of candidates for each ineffectual weight
            std::vector<weights_set> effectual_candidates (WEIGHT_LANES, weights_set());
            for(auto wgt_idx : ineffectual_weights) {
                auto lane = std::get<1>(wgt_idx);
                effectual_candidates[lane] = search(dense_filter_schedule,wgt_idx,LOOKAHEAD_H,LOOKASIDE_D);
                num_candidates[lane] = effectual_candidates[lane].size();
                min_num_candidates.push_back(effectual_candidates[lane].size());
            }

            // Promote less flexible candidates first
            overlap = min_num_candidates.empty() ? 0 : *std::min_element(min_num_candidates.begin(),
                    min_num_candidates.end());
            for(auto wgt_idx : ineffectual_weights) {
                auto lane = std::get<1>(wgt_idx);
                if(num_candidates[lane] == overlap) {
                    //Promote weight
                    auto cand_idx = effectual_candidates[lane].front();
                    promote_weight(dense_filter_schedule,wgt_idx,cand_idx);
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
    schedule BitTactical<T>::dense_scheduler(const schedule &sparse_schedule) {

        schedule dense_schedule = sparse_schedule;
        schedule result_schedule = schedule((unsigned)sparse_schedule.size(), filter_schedule());


        for(int m=0; m < sparse_schedule.size(); m++) {
            for(int time=0; time < sparse_schedule.front().size(); time++) {
                filter_scheduler(dense_schedule[m],time);
                result_schedule[m].push_back(dense_schedule[m][time]);

                //Skip zero lines
                int skip = 0;
                int next_time = time + 1;
                while (next_time < sparse_schedule.front().size() && skip < LOOKAHEAD_H &&
                        check_zero_line(dense_schedule[m][next_time])) {
                    next_time++;
                    time++;
                    skip++;
                }
            }
        }

        return result_schedule;

    }

    template <typename T>
    schedule BitTactical<T>::sparse_scheduler(const cnpy::Array<T> &wgt, int act_channels) {

        const auto &wgt_shape = wgt.getShape();

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        schedule sparse_schedule = schedule((unsigned)num_filters, filter_schedule());

        int current_group = 0, group_m = 0, start_group = 0;
        for(int m=0; m<num_filters; m++) {
            int index = 0, time = 0;
            time_schedule window_schedule;
            for (int i = 0; i < Kx; i++) {
                for (int j = 0; j < Ky; j++) {
                    for (int k = start_group; k < wgt_channels + start_group; k+=WEIGHT_LANES) {
                        for(int channel = k; channel < std::min(k + WEIGHT_LANES,act_channels); channel++) {
                            auto wgt_bits = wgt.get(m,channel - start_group, i, j);
                            window_schedule.push_back(std::make_tuple(channel,i,j,wgt_bits));
                            index++;
                            if(index == WEIGHT_LANES) {
                                sparse_schedule[m].push_back(window_schedule);
                                window_schedule.clear();
                                index = 0;
                                time++;
                            }
                        }
                    }
                }
            }

            // Ensure all the queue are equal in length
            while (index != 0) {
                window_schedule.push_back(std::make_tuple(0,0,0,0));
                index++;
                if(index == WEIGHT_LANES) {
                    sparse_schedule[m].push_back(window_schedule);
                    window_schedule.clear();
                    index = 0;
                }
            }

            group_m++;
            if(group_m >= it_per_group) {
                group_m = 0;
                current_group++;
                start_group = wgt_channels*current_group;
            }

        }

        return sparse_schedule;
    }

    template <typename T>
    schedule BitTactical<T>::scheduler(const cnpy::Array<T> &wgt, int act_channels) {
        const auto &sparse_schedule = sparse_scheduler(wgt,act_channels);
        const auto &dense_schedule = dense_scheduler(sparse_schedule);
        return dense_schedule;
    }

    /* MEMORY ACCESSES */

    template <typename T>
    void BitTactical<T>::computeMemAccessesConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void BitTactical<T>::memoryAccesses(const core::Network<T> &network) {

    }

    template class BitTactical<uint16_t>;

}