
#include <core/BitTactical.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    bool BitTactical<T>::check_schedule(const schedule &dense_schedule, int init_filter, int max_filter) {

        for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
            for(int i = 0; i < WEIGHT_LANES; i++) {
                if(!dense_schedule[filter][i].empty())
                    return true;
            }
        }
        return false;
    }

    template <typename T>
    void BitTactical<T>::update_schedule(schedule &dense_schedule, int init_filter, int max_filter) {

        for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
            for(int i = 0; i < WEIGHT_LANES; i++) {
                dense_schedule[filter][i].erase(dense_schedule[filter][i].begin());
            }
        }
    }


    /* SCHEDULER */

    void L_shape_search() {}

    void T_shape_search() {}

    void promote_weight() {}

    template <typename T>
    void BitTactical<T>::filter_scheduler(filter_schedule &sparse_filter_schedule, int filter, int time) {

        auto search = SEARCH_SHAPE == 'L' ? L_shape_search : T_shape_search;
        int overlap = 1;
        while(overlap > 0) {
            std::vector<int> num_candidates (WEIGHT_LANES, 0);

            // Get effectual weights

            // Num of candidates for each ineffectual weight

            overlap = *std::min(num_candidates.begin(), num_candidates.end());

            // Promote less flexible candidates first

        }

    }

    template <typename T>
    schedule BitTactical<T>::dense_scheduler(schedule &sparse_schedule) {

        schedule dense_schedule = schedule(sparse_schedule.size(),filter_schedule(WEIGHT_LANES,
                std::vector<std::tuple<int,int,int,T>>()));

        for(int m=0; m<sparse_schedule.size(); m++) {
            for(int time=0; time<sparse_schedule.front().front().size(); time++) {
                filter_scheduler(sparse_schedule[m],m,time);
            }
        }

        return dense_schedule;

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

        schedule naive_schedule = schedule((unsigned)num_filters, filter_schedule(WEIGHT_LANES,
                std::vector<std::tuple<int,int,int,T>>()));

        int current_group = 0, group_m = 0, start_group = 0;
        for(int m=0; m<num_filters; m++) {
            int index = 0;
            for (int i = 0; i < Kx; i++) {
                for (int j = 0; j < Ky; j++) {
                    for (int k = start_group; k < wgt_channels + start_group; k+=WEIGHT_LANES) {
                        for(int channel = k; channel < std::min(k + WEIGHT_LANES,act_channels); channel++) {
                            auto wgt_bits = wgt.get(m,channel - start_group, i, j);
                            naive_schedule[m][index].push_back(std::make_tuple(channel,i,j,wgt_bits));
                            index++;
                            if(index == WEIGHT_LANES) index = 0;
                        }
                    }
                }
            }

            // Ensure all the queue are equal in length
            while (index != 0) {
                naive_schedule[m][index].push_back(std::make_tuple(0,0,0,0));
                index++;
                if(index == WEIGHT_LANES) index = 0;
            }

            group_m++;
            if(group_m >= it_per_group) {
                group_m = 0;
                current_group++;
                start_group = wgt_channels*current_group;
            }

        }

        return naive_schedule;
    }

    template <typename T>
    schedule BitTactical<T>::scheduler(const cnpy::Array<T> &wgt, int act_channels) {
        auto sparse_schedule = sparse_scheduler(wgt,act_channels);
        //auto dense_schedule = dense_scheduler(sparse_schedule);
        return sparse_schedule;
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