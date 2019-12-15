
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Laconic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.powers_of_two_representation(data_prec);
    }

    template <typename T>
    uint8_t Laconic<T>::computeLaconicPE(uint16_t act, uint16_t wgt) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;

        #ifdef BOOTH_ENCODING
        act_bits = booth_encoding(act_bits);
        wgt_bits = booth_encoding(wgt_bits);
        #endif

        uint8_t act_effectual_bits = effectualBits(act_bits);
        uint8_t wgt_effectual_bits = effectualBits(wgt_bits);

        uint8_t bit_multiplications = act_effectual_bits * wgt_effectual_bits;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    uint8_t Laconic<T>::computeLaconicColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int init_filter, int stride, const base::Array<T> &padded_act,
            const base::Array<T> &wgt, int start_group, int max_channel, int max_filter, bool lstm, bool conv2D) {

        //Get the slowest PE
        std::vector<uint8_t> cycles;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {
            for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_channel); channel++) {

                // Fix for MobileNet
                if(conv2D)
                    start_group = filter;

                T act_bits;
                if(lstm)
                    act_bits = padded_act.get(recursion, batch, channel);
                else
                    act_bits = padded_act.get(batch, start_group + channel, stride * act_x + kernel_x,
                            stride * act_y + kernel_y);

                auto wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                uint8_t PE_cycles = computeLaconicPE(act_bits, wgt_bits);
                cycles.push_back(PE_cycles);
            }
        }

        return cycles.empty() ? 0 : *std::max_element(cycles.begin(), cycles.end());

    }

    template <typename T>
    uint8_t Laconic<T>::computeLaconicTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const base::Array<T> &padded_act, const base::Array<T> &wgt, int start_group, int max_channel,
            int max_filter, bool conv2D, uint64_t &stall_cycles) {

        //Get the slowest column
        std::vector<uint8_t> cycles;
        for(int window = 0; window < list_act_x.size(); window++) {
            uint8_t PE_cycles = computeLaconicColumn(batch,0,list_act_x[window],list_act_y[window],kernel_x,kernel_y,
                    init_channel,init_filter,stride,padded_act,wgt,start_group,max_channel,max_filter,false,conv2D);
            cycles.push_back(PE_cycles);
        }

        auto slowest_column = *std::max_element(cycles.begin(), cycles.end());
        auto fastest_column = *std::min_element(cycles.begin(), cycles.end());
        stall_cycles += slowest_column - fastest_column;
        return slowest_column;
    }

    /* CYCLES */

    template <typename T>
    void Laconic<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "Laconic_L" + std::to_string(N_LANES) + + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_T" + std::to_string(N_TILES) + "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto stall_cycles = stats.register_uint_t("stall_cycles", 0, sys::AverageTotal);
        auto weight_buff_reads = stats.register_uint_t("weight_buff_reads", 0, sys::AverageTotal);
        auto act_buff_reads = stats.register_uint_t("act_buff_reads", 0, sys::AverageTotal);
        auto accumulator_updates = stats.register_uint_t("accumulator_updates", 0, sys::AverageTotal);
        auto scheduled_pe = stats.register_uint_t("scheduled_pe", 0, sys::AverageTotal);
        auto idle_pe = stats.register_uint_t("idle_pe", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto TOTAL_ROWS = N_ROWS * N_TILES;

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.powers_of_two_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(fc) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            wgt.powers_of_two_representation(layer.getWgtPrecision());
            if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            if(act.getShape()[1] == 3 && stride > 1) {
                act.reshape_first_layer_act((uint16_t)stride);
                wgt.reshape_first_layer_wgt((uint16_t)stride);
                stride = 1;
            }

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, act_channels, Nx, Ny, R;
            if (lstm) {
                R = act_shape[0];
                batch_size = act_shape[1];
                act_channels = act_shape[2];
                Nx = 1;
                Ny = 1;
            } else {
                R = 1;
                batch_size = act_shape[0];
                act_channels = act_shape[1];
                Nx = act_shape[2];
                Ny = act_shape[3];
            }

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;;

            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_stall_cycles = 0;
                uint64_t batch_weight_buff_reads = 0;
                uint64_t batch_act_buff_reads = 0;
                uint64_t batch_accumulator_updates = 0;
                uint64_t batch_scheduled_pe = 0;
                uint64_t batch_idle_pe = 0;

                if (conv) {

                    uint64_t batch_cycles = 0;

                    std::vector<int> list_x, list_y;
                    int x_counter = 0, y_counter = 0;

                    for(int m = 0; m < num_filters; m += TOTAL_ROWS) {

                        int start_group = 0;
                        if (m >= it_per_group)
                            start_group = (int) wgt_channels;

                        bool conv2D = false;
                        if (wgt_channels == 1 && act_channels != 1)
                            conv2D = true;

                        while (iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(N_TILES, 0);
                            for (int tile = 0; tile < N_TILES; tile++) {
                                auto init_m = tile * N_ROWS + m;

                                for (int i = 0; i < Kx; i++) {
                                    for (int j = 0; j < Ky; j++) {
                                        for (int k = 0; k < wgt_channels; k += N_LANES) {

                                            auto tmp_tile_cycles = computeLaconicTile(n, list_x, list_y, i, j, k,
                                                    init_m, stride, act, wgt, start_group, (int)wgt_channels,
                                                    (int)num_filters, conv2D, batch_stall_cycles);
                                            tile_cycles[tile] += tmp_tile_cycles;

                                            batch_act_buff_reads++;
                                            batch_weight_buff_reads++;
                                            batch_scheduled_pe += list_x.size() * N_ROWS;
                                            batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                                        }
                                    }
                                }
                                batch_accumulator_updates++;
                            }
                            batch_cycles += *std::max_element(tile_cycles.begin(), tile_cycles.end());
                        }
                    }

                    cycles->value[layer_it][n] = batch_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles / N_TILES;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe;
                    idle_pe->value[layer_it][n] = batch_idle_pe;

                } else {

                    int column_index = 0;
                    std::vector<uint64_t> batch_cycles = std::vector<uint64_t>(N_TILES, 0);
                    std::vector<std::vector<uint64_t>> column_end = std::vector<std::vector<uint64_t>>(N_TILES,
                            std::vector<uint64_t>(N_COLUMNS, 0));

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m += TOTAL_ROWS) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(N_TILES, 0);
                            for (int tile = 0; tile < N_TILES; tile++) {
                                auto init_m = tile * N_ROWS + m;

                                for (int k = 0; k < wgt_channels; k += N_LANES) {
                                    if (batch_cycles[tile] < column_end[tile][column_index]) {
                                        batch_stall_cycles += column_end[tile][column_index] - batch_cycles[tile];
                                        batch_cycles[tile] = column_end[tile][column_index];
                                    }

                                    auto column_cycles = computeLaconicColumn(n, r, 0, 0, 0, 0, k, init_m, 0, act, wgt,
                                            0, (int)wgt_channels, (int)num_filters, lstm, false);

                                    column_end[tile][column_index] = batch_cycles[tile] + column_cycles;
                                    batch_cycles[tile]++;
                                    column_index++;
                                    if (column_index >= N_COLUMNS) column_index = 0;

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                }
                                batch_accumulator_updates++;
                            }
                        }
                    }

                    uint64_t max_tile_cycles = 0;
                    for (int tile = 0; tile < N_TILES; tile++) {
                        uint64_t last_column_end = *std::max_element(column_end[tile].begin(), column_end[tile].end());
                        auto tile_cycles = std::max(batch_cycles[tile], last_column_end);
                        if (tile_cycles > max_tile_cycles)
                            max_tile_cycles = tile_cycles;
                    }

                    cycles->value[layer_it][n] = max_tile_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles / N_TILES;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = (uint64_t)(num_filters * TOTAL_ROWS *
                            ceil(act_channels/(double)N_LANES));
                    auto idle_rows = TOTAL_ROWS - (num_filters % TOTAL_ROWS);
                    idle_rows = idle_rows == 16 ? 0 : idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(idle_rows * ceil(act_channels/(double)N_LANES));

                }

                act_prec->value[layer_it][n] = layer.getActPrecision();
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();

            }

        }

        //Dump statistics
        std::string header = "Laconic Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, false);

    }

    /* POTENTIALS */

    template <typename T>
    uint8_t Laconic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;

        #ifdef BOOTH_ENCODING
        act_bits = booth_encoding(act_bits);
        wgt_bits = booth_encoding(wgt_bits);
        #endif

        uint8_t act_effectual_bits = effectualBits(act_bits);
        uint8_t wgt_effectual_bits = effectualBits(wgt_bits);

        uint8_t bit_multiplications = act_effectual_bits * wgt_effectual_bits;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    std::string Laconic<T>::filename_pot() {
        return "Laconic_potentials";
    }

    template <typename T>
    std::string Laconic<T>::header_pot(const std::string &name) {
        std::string header = "Laconic Potentials/Work Reduction for " + name + "\n";
        #ifdef BOOTH_ENCODING
        header += "Booth-like Encoding\n";
        #endif
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif
        return header;
    }

    template class Laconic<uint16_t>;

}