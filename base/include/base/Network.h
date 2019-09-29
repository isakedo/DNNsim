#ifndef DNNSIM_NETWORK_H
#define DNNSIM_NETWORK_H

#include <sys/common.h>
#include <base/Layer.h>

typedef std::vector<std::vector<std::vector<std::tuple<int,int,int,uint16_t>>>> schedule;
typedef std::vector<std::vector<std::tuple<int,int,int,uint16_t>>> set_schedule;
typedef std::vector<std::tuple<int,int,int,uint16_t>> time_schedule;
typedef std::tuple<int,int,int,uint16_t> schedule_tuple;
typedef std::list<std::tuple<int,int>> weights_set;
typedef std::tuple<int,int> weight_index;

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
        uint32_t network_bits;

        /** Active forward traces */
        bool forward;

        /** Active backward traces */
        bool backward;

        /** Tensorflow 8b quantization */
        bool tensorflow_8b;

    public:

        /** Default constructor */
        Network() = default;

        /** Constructor
         * @param _name             The name of the network
         * @param _network_bits     Max number of bits of the network
         * @param _forward          Active forward traces
         * @param _backward         Active backward traces
         * @param _tensorflow_8b    Active tensorflow 8b quantization
         */
        explicit Network(const std::string &_name, uint32_t _network_bits = 16, bool _forward = false,
                bool _backward = false, bool _tensorflow_8b = false) : network_bits(_network_bits), forward(_forward),
                backward(_backward), tensorflow_8b(_tensorflow_8b) {
            name = _name;
        }

        /** Constructor
         * @param _name             The name of the network
         * @param _layers           Vector of layers
         * @param _network_bits     Max number of bits of the network
         * @param _forward          Active forward traces
         * @param _backward         Active backward traces
         * @param _tensorflow_8b    Active tensorflow 8b quantization
         */
        Network(const std::string &_name, const std::vector<Layer<T>> &_layers, uint32_t _network_bits = 16,
                bool _forward = false, bool _backward = false, bool _tensorflow_8b = false) :
                network_bits(_network_bits), forward(_forward), backward(_backward), tensorflow_8b(_tensorflow_8b) {
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
         * Get the network bits
         * @return Network bits
         */
        uint32_t getNetwork_bits() const { return network_bits; }

        /**
         * Get the if only forward traces
         * @return True if only forward traces
         */
        bool getForward() const { return forward; }

        /**
         * Get the if only backward traces
         * @return True if only backward traces
         */
        bool getBackward() const { return backward; }

        /**
         * Get the tensorflow quantization
         * @return True if tensorflow quantization
         */
        bool isTensorflow_8b() const { return tensorflow_8b; }

        /**
         * Get number of batches in the layer traces
         * @return Number of layers
         */
        uint64_t getBatches() const { return this->layers.front().getActivations().getShape()[0]; }

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
         * Set network bits
         * @param _network_bits Network bits
         */
        void setNetwork_bits(uint32_t _network_bits) { Network::network_bits = _network_bits; }

        /**
         * Update only forward flag
         * @param _forward True if only forward traces
         */
        void setForkward(bool _forward) { Network::forward = _forward; }

        /**
         * Update only backward flag
         * @param _backward True if only backward traces
         */
        void setBackward(bool _backward) { Network::backward = _backward; }

        /**
         * Update tensorflow quantization flag
         * @param _tensorflow_8b True if tensorflow quantization
         */
        void setTensorflow_8b(bool _tensorflow_8b) { Network::tensorflow_8b = _tensorflow_8b; }

        /** Return a network in fixed point given a floating point network
         * @return   Network in fixed point
         */
        Network<uint16_t> fixed_point() {
            auto fixed_network = Network<uint16_t>(name,network_bits,forward,backward,tensorflow_8b);

            for(auto &layer : layers) {
                auto fixed_layer = Layer<uint16_t>(layer.getType(),layer.getName(),layer.getInput(),layer.getNn(),
                        layer.getKx(),layer.getKy(),layer.getStride(),layer.getPadding(),layer.getActPrecision(),
                        layer.getActMagnitude(),layer.getActFraction(),layer.getWgtPrecision(),layer.getWgtMagnitude(),
                        layer.getWgtFraction());

                if(tensorflow_8b) fixed_layer.setActivations(layer.getActivations().tensorflow_fixed_point());
                else fixed_layer.setActivations(layer.getActivations().profiled_fixed_point(layer.getActMagnitude(),
                        layer.getActFraction()));
                layer.setActivations(Array<T>());

                if(tensorflow_8b) fixed_layer.setWeights(layer.getWeights().tensorflow_fixed_point());
                else fixed_layer.setWeights(layer.getWeights().profiled_fixed_point(layer.getWgtMagnitude(),
                        layer.getWgtFraction()));
                layer.setWeights(Array<T>());

                fixed_network.updateLayers().emplace_back(fixed_layer);
            }

            return fixed_network;
        }

        /** Duplicate the decoder layers to store all decode steps
         * @param decoder_states Number of decoder states in the traces
         */
        void duplicate_decoder_layers(int decoder_states) {

            std::vector<Layer<T>> tmp_layers;
            std::vector<Layer<T>> tmp_decoders;

            for(const auto layer : this->layers) {
                if(layer.getType() == "Decoder") tmp_decoders.push_back(layer);
                else tmp_layers.push_back(layer);
            }

            for(int decoder_state = 0; decoder_state < decoder_states; decoder_state++) {
                for(const auto layer : tmp_decoders) {
                    tmp_layers.push_back(layer);
                    tmp_layers.back().setName(tmp_layers.back().getName() + "_" + std::to_string(decoder_state));
                }
            }

            this->layers = tmp_layers;

        }

    };

}

#endif //DNNSIM_NETWORK_H
