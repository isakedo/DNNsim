#ifndef DNNSIM_NETWORK_H
#define DNNSIM_NETWORK_H

#include <base/Layer.h>

namespace base {

    /**
     * Container for the network
     * @tparam T Data type of the network
     */
    template <typename T>
    class Network {

    private:

        /** Name of the network */
        std::string name;

        /** Set of layers of the network*/
        std::vector<Layer<T>> layers;

        /** Max number of bits for the network*/
        uint32_t data_width = 0;

        /** Profiled quantization */
        bool profiled = false;

    public:

        /** Default constructor */
        Network() = default;

        /** Constructor
         * @param _name             The name of the network
         * @param _data_width       Max number of bits of the network
         * @param _profiled         Active profiled quantization
         */
        explicit Network(const std::string &_name, uint32_t _data_width = 16, bool _profiled = false) :
                data_width(_data_width), profiled(_profiled) {
            name = _name;
        }

        /** Constructor
         * @param _name             The name of the network
         * @param _layers           Vector of layers
         * @param _data_width       Max number of bits of the network
         * @param _profiled         Active profiled quantization
         */
        Network(const std::string &_name, const std::vector<Layer<T>> &_layers, uint32_t _data_width = 16,
                bool _profiled = false) : data_width(_data_width), profiled(_profiled) {
            name = _name; layers = _layers;
        }

        /**
         * Get name of the network
         * @return Name of the network
         */
        const std::string &getName() const { return name; }

        /**
         * Get the layers data
         * @return Array of layers
         */
        const std::vector<Layer<T>> &getLayers() const { return layers; }

        /**
         * Get the network width
         * @return Network width
         */
        uint32_t getNetworkWidth() const { return data_width; }

        /**
         * Get batch soze in the layer traces
         * @return Btach
         */
        uint64_t getBatchSize() const {
            uint64_t max_batch_size = 0;
            for (const auto &layer : this->layers) {
                uint64_t batch_size = layer.getActivations().getShape()[0];
                if (batch_size > max_batch_size)
                    max_batch_size = batch_size;
            }
            return max_batch_size;
        }

        /**
         * Get number of layers in the network
         * @return Number of layers in the network
         */
        uint64_t getNumLayers() const { return this->layers.size(); }

        /**
         * Get the names for the layers of the network
         * @return Array with the name of the layers
         */
        std::vector<std::string> getLayersName() const {
            std::vector<std::string> layers_name;
            for(const auto &layer : layers) { layers_name.push_back(layer.getName()); }
            return layers_name;
        }

        /**
         * Get reference to the layers
         * @return Pointer to the layers
         */
        std::vector<Layer<T>> &updateLayers() { return layers; }

        /**
         * Set network width
         * @param _network_width Network width
         */
        void setNetworkWidth(uint32_t _network_width) { Network::data_width = _network_width; }

        /**
         * Update profiled quantization flag
         * @param _profiled True if profiled quantization
         */
        void setProfiled(bool _profiled) { Network::profiled = _profiled; }

        /** Return a network in fixed point given a floating point network
         * @return   Network in fixed point
         */
        Network<uint16_t> fixed_point() {
            auto fixed_network = Network<uint16_t>(name, data_width, profiled);

            for(auto &layer : layers) {
                auto fixed_layer = Layer<uint16_t>(layer.getName(), layer.getType(), layer.getStride(),
                        layer.getPadding(), layer.getActPrecision(), layer.getActMagnitude(), layer.getActFraction(),
                        layer.getWgtPrecision(), layer.getWgtMagnitude(), layer.getWgtFraction());

                if (profiled) fixed_layer.setActivations(layer.getActivations().profiled_quantization(layer.getActMagnitude(),
                        layer.getActFraction()));
                else fixed_layer.setActivations(layer.getActivations().linear_quantization(data_width));
                layer.setActivations(Array<T>()); // Clear

                if (profiled) fixed_layer.setWeights(layer.getWeights().profiled_quantization(layer.getWgtMagnitude(),
                        layer.getWgtFraction()));
                else fixed_layer.setWeights(layer.getWeights().linear_quantization(data_width));
                layer.setWeights(Array<T>()); // Clear

                fixed_network.updateLayers().emplace_back(fixed_layer);
            }

            return fixed_network;
        }

    };

}

#endif //DNNSIM_NETWORK_H
