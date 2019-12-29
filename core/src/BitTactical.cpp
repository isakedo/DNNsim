
#include <core/BitTactical.h>

namespace core {

    /* SCHEDULER */

    template<typename T>
    uint32_t BitTactical<T>::getLookaheadH() const {
        return LOOKAHEAD_H;
    }

    template <typename T>
    bool BitTactical<T>::check_zero_line(const BufferRow<T> &buffer) {
        for(auto tuple : buffer) {
            auto value = std::get<0>(tuple);
            if(value != 0) return false;
        }
        return true;
    }

    template <typename T>
    void BitTactical<T>::promote(BufferSet<T> &buffer, ValueIndex ineffectual, ValueIndex candidate) {

        // Ineffectual
        auto inef_time = std::get<0>(ineffectual);
        auto inef_lane = std::get<1>(ineffectual);

        // Candidate
        auto cand_time = std::get<0>(candidate);
        auto cand_lane = std::get<1>(candidate);

        // Swap
        auto ineffectual_tuple = buffer[inef_time][inef_lane];
        buffer[inef_time][inef_lane] = buffer[cand_time][cand_lane];
        buffer[cand_time][cand_lane] = ineffectual_tuple;
    }

    template <typename T>
    std::vector<ValueIndex> BitTactical<T>::search(const BufferSet<T> &buffer, ValueIndex value_idx,
            int max_time) {

        auto time = std::get<0>(value_idx);
        auto lane = std::get<1>(value_idx);
        int upper_bound = (lane / N_LANES) * N_LANES;
        int lower_bound = ((lane / N_LANES) + 1) * N_LANES;
        std::vector<ValueIndex> effectual_candidates;

        auto next_time = time + 1;
        if(next_time >= max_time) return effectual_candidates;

        // Search effectual values in search space
        for (int s = 0; s < SEARCH_MAP.size(); ++s) {
            auto search_space = SEARCH_MAP[s];
            auto time_h = time + std::get<0>(search_space);
            auto lane_d = lane + std::get<1>(search_space);
            if(time_h >= max_time) continue;
            lane_d = (lane_d) < upper_bound ? N_LANES + lane_d : lane_d; // Wrap around
            lane_d = (lane_d) >= lower_bound ? lane_d - N_LANES : lane_d; // Wrap around
            auto value_tuple = buffer[time_h][lane_d];
            auto value_bits = std::get<0>(value_tuple);
            if(value_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h, lane_d));
        }

        return effectual_candidates;
    }

    template <typename T>
    void BitTactical<T>::original_schedule(BufferSet<T> &buffer) {

        auto max_time = buffer.size();
        auto groups = buffer.front().size() / N_LANES;

        int skip = 0;
        for (int time = 0; time < max_time; ++time) {

            // Skip lines of zeroes
            if (skip < LOOKAHEAD_H && check_zero_line(buffer[time])) {
                skip++;
                continue;
            }
            skip = 0;

            for (int group = 0; group < groups; ++group) {

                int overlap = 1;
                while(overlap > 0) {

                    // Get ineffectual values
                    int init_lane = group * N_LANES;
                    std::vector<ValueIndex> ineffectual_values;
                    for(int lane = init_lane; lane < init_lane + N_LANES; lane++) {
                        auto value_tuple = buffer[time][lane];
                        auto value_bits = std::get<0>(value_tuple);
                        if(value_bits == 0) ineffectual_values.emplace_back(std::make_tuple(time, lane));
                    }

                    // Num of candidates for each ineffectual values
                    overlap = -1;
                    std::vector<uint16_t> num_candidates (N_LANES, 0);
                    std::vector<std::vector<ValueIndex>> effectual_candidates (N_LANES, std::vector<ValueIndex>());
                    for(auto inef_idx : ineffectual_values) {
                        auto lane = std::get<1>(inef_idx);
                        effectual_candidates[lane % N_LANES] = search(buffer, inef_idx, max_time);
                        if(!effectual_candidates[lane % N_LANES].empty()) {
                            auto effectual_num_candidates = (uint16_t)effectual_candidates[lane % N_LANES].size();
                            num_candidates[lane % N_LANES] = effectual_num_candidates;
                            if (effectual_num_candidates > overlap) overlap = effectual_num_candidates;
                        }
                    }

                    // Promote less flexible candidates first
                    for(auto inef_idx : ineffectual_values) {
                        auto lane = std::get<1>(inef_idx);
                        if(num_candidates[lane % N_LANES] == overlap) {
                            //Promote value
                            auto cand_idx = effectual_candidates[lane % N_LANES].front();
                            promote(buffer, inef_idx, cand_idx);
                            break;
                        }
                    }

                } // Optimal promotion loop

            } // Group
        } // Time

    }

    template <typename T>
    void BitTactical<T>::schedule(Buffer<T> &buffer, uint32_t _N_LANES) {
        N_LANES = _N_LANES;
        for (auto &buffer_set : buffer) {
            original_schedule(buffer_set);
        }
    }

    INITIALISE_DATA_TYPES(BitTactical);

}