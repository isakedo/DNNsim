
#include <core/BitTacticalP.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPBitsPE(uint16_t wgt, uint8_t act_layer_prec, const int network_bits) {
        #ifdef ZERO_COUNT
        return wgt == 0 ? (uint8_t)1 : act_layer_prec * (uint8_t)network_bits;
        #else
        return wgt == 0 ? (uint8_t)0 : act_layer_prec * (uint8_t)NETWORK_BITS;
        #endif
    }

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPColumn(int batch, int recursion, int act_x, int act_y, int stride,
            const cnpy::Array<T> &padded_act, const schedule &dense_schedule, int schedule_time, uint16_t act_mask,
            bool lstm) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int row = 0; row < this->N_ROWS; row++) {
            for (int wgt_idx = 0; wgt_idx < this->N_LANES; wgt_idx++) {

                int pos = row * this->N_LANES + wgt_idx;
                auto wgt_tuple = dense_schedule[schedule_time][pos];
                int channel = std::get<0>(wgt_tuple);
                int kernel_x = std::get<1>(wgt_tuple);
                int kernel_y = std::get<2>(wgt_tuple);
                if(channel < 0) continue;

                // Computation cycles
                uint16_t act_bits;
                if(lstm)
                    act_bits = padded_act.get(recursion, batch, channel);
                else
                    act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);

                bool neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~act_mask;
                    neg = true;
                }

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if(neg) max_act_bit += 1;

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }
        }

        int cycles;
        if(!LEADING_BIT) cycles = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        else cycles = max_bit + 1;
        return (uint8_t)cycles;
    }

    template <typename T>
    void BitTacticalP<T>::computeTacticalPTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int stride, const cnpy::Array<T> &padded_act,
            const schedule &dense_schedule, int schedule_time, uint16_t act_mask,
            std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
            sys::Statistics::Stats &stats) {

        uint32_t N_GROUPS = this->N_COLUMNS * 16 / PRECISION_GRANULARITY;
        uint32_t WINDOWS_PER_GROUP = this->N_COLUMNS / N_GROUPS;

        std::vector<uint8_t> per_group_cycles (N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for (int row = 0; row < this->N_ROWS; row++) {
                for (int wgt_idx = 0; wgt_idx < this->N_LANES; wgt_idx++) {

                    int pos = row * this->N_LANES + wgt_idx;
                    auto wgt_tuple = dense_schedule[schedule_time][pos];
                    int channel = std::get<0>(wgt_tuple);
                    int kernel_x = std::get<1>(wgt_tuple);
                    int kernel_y = std::get<2>(wgt_tuple);
                    if(channel < 0) continue;

                    // Computation cycles
                    uint16_t act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);

                    bool neg = false;
                    if((act_bits & act_mask) != 0) {
                        act_bits = act_bits & ~act_mask;
                        neg = true;
                    }

                    const auto &min_max_act_bits = this->minMax(act_bits);

                    auto min_act_bit = std::get<0>(min_max_act_bits);
                    auto max_act_bit = std::get<1>(min_max_act_bits);

                    if(neg) max_act_bit += 1;

                    if(min_act_bit < min_bit) min_bit = min_act_bit;
                    if(max_act_bit > max_bit) max_bit = max_act_bit;

                }
            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 :
                        max_bit - min_bit + 1);
                else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
            else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);
        }

        for(int group = 0; group < N_GROUPS; group++) {
            cycles_per_group[group] += per_group_cycles[group];
        }

        if(this->COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_group) {
                if(column_cycles <= end_previous_pallet[0]) {
                    if(column_cycles < fastest_column) fastest_column = column_cycles;
                    column_cycles = end_previous_pallet[0] + 1;
                }
            }
            stats.stall_cycles.back()[batch] += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < this->COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[this->COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_group.begin(),
                    cycles_per_group.end());
        } else {
            auto slowest_group = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
            auto fastest_group = *std::min_element(cycles_per_group.begin(), cycles_per_group.end());
            cycles_per_group = std::vector<uint32_t>(N_GROUPS, slowest_group);
            stats.stall_cycles.back()[batch] += slowest_group - fastest_group;
        }

    }

    /* CYCLES */

    template <typename T>
    void BitTacticalP<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const schedule &proto_dense_schedule) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);

        if(act.getShape()[1] == 3 && stride > 1) {
            act.reshape_first_layer_act((uint16_t)stride);
            wgt.reshape_first_layer_wgt((uint16_t)stride);
            stride = 1;
        }

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        auto act_prec = layer.getActPrecision();
        auto act_mask = (uint16_t)(1u << (act_prec - 1));

        auto groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)this->N_ROWS/groups);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.scheduled_pe.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        schedule dense_schedule;
        if(proto_dense_schedule.empty())
            dense_schedule = this->scheduler(wgt,act_channels);
        else
            dense_schedule = proto_dense_schedule;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(this->COLUMN_REGISTERS, 0);
            std::vector<uint32_t> cycles_per_group = std::vector<uint32_t>(this->N_COLUMNS * 16 /
                    PRECISION_GRANULARITY, 0);
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;
            uint64_t scheduled_pe = 0;
            uint64_t idle_pe = 0;

            while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, this->N_COLUMNS)) {
                for(int schedule_time = 0; schedule_time < dense_schedule.size(); schedule_time++) {
                    computeTacticalPTile(n, list_x, list_y, stride, act, dense_schedule, schedule_time, act_mask,
                            cycles_per_group, end_previous_pallet, stats);

                    act_buff_reads++;
                    weight_buff_reads++;
                    scheduled_pe += list_x.size() * this->N_ROWS;
                    idle_pe += (this->N_COLUMNS - list_x.size()) * this->N_ROWS;
                }
                accumulator_updates++;
            }

            stats.cycles.back()[n] = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates * num_filters_sets;
            stats.scheduled_pe.back()[n] = scheduled_pe;
            stats.idle_pe.back()[n] = idle_pe;

        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void BitTacticalP<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const schedule &proto_dense_schedule) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.reshape_to_4D();

        if(layer.getType() == "InnerProduct") {
            if (act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        uint64_t batch_size, act_channels, R;
        if(lstm) {
            R = act_shape[0];
            batch_size = act_shape[1];
            act_channels = act_shape[2];
        } else {
            R = 1;
            batch_size = act_shape[0];
            act_channels = act_shape[1];
        }
        if(this->FAST_MODE) batch_size = 1;

        auto num_filters = wgt_shape[0];

        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)this->N_ROWS);

        auto act_prec = layer.getActPrecision();
        auto act_mask = (uint16_t)(1u << (act_prec - 1));

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.scheduled_pe.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        schedule dense_schedule;
        if(proto_dense_schedule.empty())
            dense_schedule = this->scheduler(wgt,act_channels);
        else
            dense_schedule = proto_dense_schedule;

        #ifndef FC_MULTIPLEX_COLUMNS

        // All FC in one column
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {

            uint64_t cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;

            for (int r = 0; r < R; r++) {
                for(int schedule_time = 0; schedule_time < dense_schedule.size(); schedule_time++) {
                   cycles += computeTacticalPColumn(n,r,0,0,0,act,dense_schedule,schedule_time,act_mask,lstm);
                    weight_buff_reads++;
                    act_buff_reads++;
                }
                accumulator_updates++;
            }

            stats.cycles.back()[n] = cycles;
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates;
            stats.scheduled_pe.back()[n] = num_filters * this->N_ROWS * ceil(act_channels/(double)WEIGHT_LANES);
            auto idle_rows = this->N_ROWS - (num_filters % this->N_ROWS);
            idle_rows = idle_rows == 16 ? 0 : idle_rows;
            stats.idle_pe.back()[n] = idle_rows * ceil(act_channels/(double)WEIGHT_LANES);

        }

        #else

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {

            int column_index = 0;
            std::vector<uint64_t> column_end = std::vector<uint64_t>(this->N_COLUMNS, 0);
            uint64_t cycles = 0;
            uint64_t stall_cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;

            for (int r = 0; r < R; r++) {
                for(int schedule_time = 0; schedule_time < dense_schedule.size(); schedule_time++) {
                    if(cycles < column_end[column_index]) {
                        stall_cycles = column_end[column_index] - cycles;
                        cycles = column_end[column_index];
                    }
                    auto column_cycles = computeTacticalPColumn(n,r,0,0,0,act,dense_schedule,schedule_time,act_mask,
                            lstm);
                    column_end[column_index] = cycles + column_cycles;
                    cycles++;
                    column_index++;
                    if(column_index >= this->N_COLUMNS) column_index = 0;

                    weight_buff_reads++;
                    act_buff_reads++;
                }
                accumulator_updates++;
            }

            uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
            stats.cycles.back()[n] = std::max(cycles, last_column_end);
            stats.stall_cycles.back()[n] = stall_cycles;
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates * num_filters_sets;
            stats.scheduled_pe.back()[n] = uint64_t(num_filters * this->N_ROWS *
                    ceil(act_channels/(double)this->N_LANES));
            auto idle_rows = this->N_ROWS - (num_filters % this->N_ROWS);
            idle_rows = idle_rows == 16 ? 0 : idle_rows;
            stats.idle_pe.back()[n] = (uint64_t)(idle_rows * ceil(act_channels/(double)this->N_LANES));

        }

        #endif

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void BitTacticalP<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        int mux_entries = this->LOOKAHEAD_H + this->LOOKASIDE_D + 1;
        stats.arch = "BitTacticalP_C" + std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS) + "_PG_" +
                std::to_string(PRECISION_GRANULARITY) + "_CR" + std::to_string(this->COLUMN_REGISTERS) + "_" +
                this->SEARCH_SHAPE + std::to_string(mux_entries) + "(" + std::to_string(this->LOOKAHEAD_H) + "-" +
                std::to_string(this->LOOKASIDE_D) + ")" + (LEADING_BIT ? "_LB" : "");

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                computeConvolution(layer, stats, schedule());
            } else if(layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                computeInnerProduct(layer, stats, schedule());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template <typename T>
    void BitTacticalP<T>::run(const Network<T> &network, const std::vector<schedule> &schedules) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        int mux_entries = this->LOOKAHEAD_H + this->LOOKASIDE_D + 1;
        stats.arch = "BitTacticalP_C" + std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS) + "_PG_" +
                std::to_string(PRECISION_GRANULARITY) + "_CR" + std::to_string(this->COLUMN_REGISTERS) + "_" +
                this->SEARCH_SHAPE + std::to_string(mux_entries) + "(" + std::to_string(this->LOOKAHEAD_H) + "-" +
                std::to_string(this->LOOKASIDE_D) + ")" + (LEADING_BIT ? "_LB" : "");

        int sch_index = 0;
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                const schedule &proto_dense_schedule = schedules[sch_index];
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                computeConvolution(layer, stats, proto_dense_schedule);
                sch_index++;
            } else if(layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                const schedule &proto_dense_schedule = schedules[sch_index];
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                computeInnerProduct(layer, stats, proto_dense_schedule);
                sch_index++;
            }
        }
        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void BitTacticalP<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        auto batch_size = act_shape[0];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

        int n;

        // Get layer precision
        auto act_layer_prec = layer.getActPrecision();

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < batch_size; n++) {
            uint64_t bit_counter = 0;
            for(int m=0; m < num_filters; m++) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < wgt_channels; k++) {
                            bit_counter += computeTacticalPBitsPE(wgt.get(m, k, i, j), (uint8_t)act_layer_prec,
                                    network_bits);
                        }
                    }
                }
            }
            bit_counter *= out_x * out_y;
            double MAX_BITS = network_bits * network_bits;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
            stats.speedup.back()[n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void BitTacticalP<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        auto batch_size = act_shape[0];
        auto R = (layer.getType() == "LSTM") ? act_shape[0] : 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels * R;
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

        int n;

        // Get layer precision
        auto act_layer_prec = layer.getActPrecision();

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {
            uint64_t bit_counter = 0;
            for (int r = 0; r < R; r++) {
                for (int m = 0; m < num_filters; m++) {
                    for (int k = 0; k < wgt_channels; k++) {
                        bit_counter += computeTacticalPBitsPE(wgt.get(m, k), (uint8_t) act_layer_prec, network_bits);
                    }
                }
            }
            double MAX_BITS = network_bits * network_bits;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
            stats.speedup.back()[n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void BitTacticalP<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "BitTacticalP";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(0);
                computePotentialsConvolution(layer,stats,network.getNetwork_bits());
            } else if (layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats,network.getNetwork_bits());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class BitTacticalP<uint16_t>;

}

