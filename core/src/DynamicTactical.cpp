
#include <core/DynamicTactical.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void DynamicTactical<T>::run(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicTactical";
        std::string filename = arch + "_cycles";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats

        // Backward stats

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode, true);

            // Forward pass
            if (simulate.only_forward || !simulate.only_backward) {

                if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles forward simulation for epoch "
                        << epoch << std::endl;

                for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                    const base::Layer<float> &layer = network.getLayers()[layer_it];


                }
            }

            // Backward pass
            if (simulate.only_backward || !simulate.only_forward) {

                if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles backward simulation for epoch "
                        << epoch << std::endl;

                for (int layer_it = network.getNumLayers() - 1; layer_it >= 0; layer_it--) {

                    const base::Layer<float> &layer = network.getLayers()[layer_it];


                }
            }

        }

        //Dump statistics
        std::string header = "DynamicTactical Number of Cycles for " + network_model.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    /* CYCLES */

    template <typename T>
    void DynamicTactical<T>::potentials(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicTactical";
        std::string filename = arch + "_cycles";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats

        // Backward stats

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode, true);

            // Forward pass
            if (simulate.only_forward || !simulate.only_backward) {

                if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles forward simulation for epoch "
                                           << epoch << std::endl;

                for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                    const base::Layer<float> &layer = network.getLayers()[layer_it];


                }
            }

            // Backward pass
            if (simulate.only_backward || !simulate.only_forward) {

                if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles backward simulation for epoch "
                                           << epoch << std::endl;

                for (int layer_it = network.getNumLayers() - 1; layer_it >= 0; layer_it--) {

                    const base::Layer<float> &layer = network.getLayers()[layer_it];


                }
            }

        }

        //Dump statistics
        std::string header = "DynamicTactical Potentials for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    template class DynamicTactical<float>;

}
