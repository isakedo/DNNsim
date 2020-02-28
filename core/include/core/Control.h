#ifndef DNNSIM_CONTROL_H
#define DNNSIM_CONTROL_H

#include "Utils.h"
#include "BitTactical.h"
#include "DRAM.h"
#include "GlobalBuffer.h"
#include "Architecture.h"

namespace core {

    /**
     * Control Logic
     * @tparam T Data type values
     */
    template <typename T>
    class Control {

    protected:

        class Node {
        public:
            std::vector<AddressRange> read_addresses;
            std::vector<AddressRange> evict_addresses;
            std::vector<AddressRange> write_addresses;
        };

        std::vector<std::shared_ptr<Node>> on_chip_graph;

        /** Weight buffer scheduler */
        std::shared_ptr<BitTactical<T>> scheduler;

        std::shared_ptr<DRAM<T>> dram;

        std::shared_ptr<GlobalBuffer<T>> gbuffer;

        std::shared_ptr<Architecture<T>> arch;

        /** Pointer to activations */
        std::shared_ptr<base::Array<T>> act;

        /** Pointer to weights */
        std::shared_ptr<base::Array<T>> wgt;

        /** Indicate if linear layer (alternate fashion window buffer) */
        bool linear = false;

        /** Indicate if LSTM layer (different dimensions order) */
        bool lstm = false;

        /** Stride of the layer */
        int stride = 0;

        /** Number of effective concurrent multiplications per PE */
        uint32_t EF_LANES = 0;

        /** Number of effective columns */
        uint32_t EF_COLUMNS = 0;

        /** Number of efffective rows */
        uint32_t EF_ROWS = 0;

        bool layer_act_on_chip = false;

        bool next_layer_act_on_chip = false;

        virtual void generate_memory_maps() = 0;

        virtual void generate_execution_graph() = 0;

    public:

        /**
         * Constructor
         * @param _scheduler
         * @param _dram
         * @param _gbuffer
         * @param _arch
         */
        Control(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<Architecture<T>> &_arch) :
                scheduler(_scheduler), dram(_dram), gbuffer(_gbuffer), arch(_arch) {}

        const std::shared_ptr<DRAM<T>> &getDram() const {
            return dram;
        }

        const std::shared_ptr<GlobalBuffer<T>> &getGbuffer() const {
            return gbuffer;
        }

        const std::shared_ptr<Architecture<T>> &getArch() const {
            return arch;
        }

        /**
        * Return name for the dataflow
        * @return Name
        */
        virtual std::string name() = 0;

        /**
         * Configure control values for the current layer
         * @param _act          Activation array
         * @param _wgt          Weight array
         * @param _linear       Linear layer
         * @param _lstm         LSTM
         * @param _stride       Stride
         * @param _EF_COLUMNS   Number of effective columns
         * @param _EF_ROWS      Number of effective rows
         */
        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _linear, bool _lstm, int _stride,
                uint32_t _EF_COLUMNS, uint32_t _EF_ROWS) {
            act = _act;
            wgt = _wgt;
            linear = _linear;
            lstm = _lstm;

            stride = _stride;
            EF_LANES = arch->getNLanes();
            EF_COLUMNS = _EF_COLUMNS;
            EF_ROWS = _EF_ROWS;

            layer_act_on_chip = next_layer_act_on_chip;
            next_layer_act_on_chip = false;
        }

        const std::vector<AddressRange> &getReadAddresses() {
            return on_chip_graph.front()->read_addresses;
        }

        const std::vector<AddressRange> &getEvictAddresses() {
            return on_chip_graph.front()->evict_addresses;
        }

        bool still_off_chip_data() {
            on_chip_graph.erase(on_chip_graph.begin());
            return !on_chip_graph.empty();
        }

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        virtual bool still_on_chip_data(std::vector<TileData<T>> &tiles_data) = 0;

    };

}

#endif //DNNSIM_CONTROL_H
