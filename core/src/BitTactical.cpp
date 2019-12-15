
#include <core/BitTactical.h>
#include <iomanip>

namespace core {

    /* SCHEDULER */

    void promote_weight(set_schedule &dense_filter_schedule, weight_index ineffectual, weight_index candidate) {
        auto inef_time = std::get<0>(ineffectual);
        auto inef_lane = std::get<1>(ineffectual);
        auto cand_time = std::get<0>(candidate);
        auto cand_lane = std::get<1>(candidate);
        auto ineffectual_tuple = dense_filter_schedule[inef_time][inef_lane];
        dense_filter_schedule[inef_time][inef_lane] = dense_filter_schedule[cand_time][cand_lane];
        dense_filter_schedule[cand_time][cand_lane] = ineffectual_tuple;
    }

    template <typename T>
    weights_set BitTactical<T>::weight_search(const set_schedule &dense_schedule, weight_index wgt_idx, int max_time) {

        auto time = std::get<0>(wgt_idx);
        auto lane = std::get<1>(wgt_idx);
        auto upper_bound = (lane/(int)N_LANES)*(int)N_LANES;
        auto lower_bound = ((lane/(int)N_LANES)+1)*(int)N_LANES;
        weights_set effectual_candidates;
        auto next_time = time + 1;
        if(next_time >= max_time) return effectual_candidates;

        for (auto search_space : SEARCH_MAP) {
            auto time_h = time + std::get<0>(search_space);
            auto lane_d = lane + std::get<1>(search_space);
            if(time_h >= max_time) continue;
            lane_d = (lane_d) < upper_bound ? N_LANES + lane_d : lane_d; // Wrap around
            lane_d = (lane_d) >= lower_bound ? lane_d - N_LANES : lane_d; // Wrap around
            auto wgt_tuple = dense_schedule[time_h][lane_d];
            auto wgt_bits = std::get<3>(wgt_tuple);
            if(wgt_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h,lane_d));
        }

        return effectual_candidates;
    }

    template <typename T>
    void BitTactical<T>::filter_scheduler(set_schedule &dense_schedule, int time, int row, int max_time) {

        int overlap = 1;
        while(overlap > 0) {

            std::vector<int> num_candidates (N_ROWS * N_LANES, 0);
            std::vector<int> min_num_candidates;

            // Get ineffectual weights
            int init_lane = row * N_LANES;
            weights_set ineffectual_weights;
            for(int lane = init_lane; lane < init_lane + N_LANES; lane++) {
                auto wgt_tuple = dense_schedule[time][lane];
                auto wgt_bits = std::get<3>(wgt_tuple);
                if(wgt_bits == 0) ineffectual_weights.push_back(std::make_tuple(time,lane));
            }

            // Num of candidates for each ineffectual weight
            std::vector<weights_set> effectual_candidates (N_ROWS * N_LANES, weights_set());
            for(auto wgt_idx : ineffectual_weights) {
                auto lane = std::get<1>(wgt_idx);
                effectual_candidates[lane] = weight_search(dense_schedule,wgt_idx,max_time);
                if(!effectual_candidates[lane].empty()) {
                    num_candidates[lane] = (int)effectual_candidates[lane].size();
                    min_num_candidates.push_back((int)effectual_candidates[lane].size());
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
    schedule BitTactical<T>::dense_scheduler(const schedule &sparse_schedule) {

        schedule result_schedule = schedule();

        for(const auto &set_sparse_schedule : sparse_schedule) {

            set_schedule dense_schedule = set_sparse_schedule;
            set_schedule set_result_schedule = set_schedule();

            int skip = 0;
            int max_time = set_sparse_schedule.size();
            for (int time = 0; time < max_time; time++) {

                //Skip zero lines
                if (skip < LOOKAHEAD_H && check_zero_line(dense_schedule[time])) {
                    skip++;
                    continue;
                }

                skip = 0;

                // Divide in filters to process faster
                for (int row = 0; row < N_ROWS; row++) {
                    filter_scheduler(dense_schedule, time, row, max_time);
                }

                set_result_schedule.emplace_back(dense_schedule[time]);
            }
            result_schedule.emplace_back(set_result_schedule);
        }

        return result_schedule;

    }

    template <typename T>
    schedule BitTactical<T>::sparse_scheduler(const base::Array<T> &wgt, uint64_t act_channels, bool fc) {

        const auto &wgt_shape = wgt.getShape();

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto groups = act_channels / wgt_channels;
        auto it_per_group = num_filters / groups;
        int round_wgt_channels = (int)ceil(wgt_channels/(double)N_LANES)*N_LANES;

        auto filters_per_set = std::min(N_ROWS, (uint32_t)ceil(num_filters/(double)1));
        auto num_filter_sets = (uint64_t)ceil(num_filters/(double)filters_per_set);
        auto time_per_filter = (uint64_t)ceil(round_wgt_channels*Kx*Ky/(double)N_LANES);

        schedule sparse_schedule = schedule(num_filter_sets, set_schedule(time_per_filter,
                time_schedule(N_ROWS * N_LANES, schedule_tuple(-1,-1,-1,0))));

        int set = -1;
        for(int m = 0; m < num_filters; m++) {

            if ((m % filters_per_set) == 0)
                set++;

            // Two towers alexnet
            int start_group = 0;
            if(m >= it_per_group)
                start_group = (int)wgt_channels;

            // Fix for MobileNet
            if(wgt_channels == 1 && act_channels != 1)
                start_group = m;

            int time = 0;
            for (int i = 0; i < Kx; i++) {
                for (int j = 0; j < Ky; j++) {
                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        int index = 0;
                        for(int channel = k; channel < std::min(k + (int)N_LANES,(int)wgt_channels); channel++) {
                            auto wgt_bits = fc ? wgt.get(m, channel) : wgt.get(m,channel,i,j);
                            int pos = (m % filters_per_set) * N_LANES + index;
                            sparse_schedule[set][time][pos] = std::make_tuple(start_group + channel,i,j,wgt_bits);
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

        }

        return sparse_schedule;
    }

    template <typename T>
    schedule BitTactical<T>::scheduler(const base::Array<T> &wgt, uint64_t act_channels, bool fc) {
        const auto &sparse_schedule = sparse_scheduler(wgt,act_channels, fc);
        const auto &dense_schedule = dense_scheduler(sparse_schedule);
        return dense_schedule;
    }

    template <typename T>
    std::vector<schedule> BitTactical<T>::network_scheduler(const base::Network<T> &network) {

        std::vector<schedule> network_schedule;
        for(const base::Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {

                base::Array<T> act = layer.getActivations();
                base::Array<T> wgt = layer.getWeights();
                if (wgt.getDimensions() == 2) wgt.reshape_to_4D();

                auto stride = layer.getStride();

                if(wgt.getShape()[1] == 3 && stride > 1) {
                    act.reshape_first_layer_act((uint16_t) stride);
                    wgt.reshape_first_layer_wgt((uint16_t) stride);
                }

                const auto &dense_schedule = scheduler(wgt, act.getShape()[1], false);
                network_schedule.push_back(dense_schedule);

            } else if(layer.getType() == "InnerProduct") {

                const base::Array<T> &wgt = layer.getWeights();
                const auto &dense_schedule = scheduler(wgt, layer.getWeights().getShape()[1], true);
                network_schedule.push_back(dense_schedule);

            }
        }

        return network_schedule;

    }

    template class BitTactical<uint16_t>;

}