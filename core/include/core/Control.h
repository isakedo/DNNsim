#ifndef DNNSIM_CONTROL_H
#define DNNSIM_CONTROL_H

#include "Utils.h"
#include "BitTactical.h"
#include "DRAM.h"
#include "GlobalBuffer.h"
#include "LocalBuffer.h"
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
            bool evict_act = false;
            bool evict_wgt = false;
            std::vector<AddressRange> read_act_addresses;
            std::vector<AddressRange> read_wgt_addresses;
            std::vector<AddressRange> write_addresses;
        };

        std::vector<std::shared_ptr<Node>> on_chip_graph;

        /** Weight buffer scheduler */
        std::shared_ptr<BitTactical<T>> scheduler;

        std::shared_ptr<DRAM<T>> dram;

        std::shared_ptr<GlobalBuffer<T>> gbuffer;

        std::shared_ptr<LocalBuffer<T>> abuffer;

        std::shared_ptr<LocalBuffer<T>> wbuffer;

        std::shared_ptr<LocalBuffer<T>> obuffer;

        // TODO: Composer column

        // TODO: Activation function unit

        std::shared_ptr<Architecture<T>> arch;

        /** Pointer to activations */
        std::shared_ptr<base::Array<T>> act;

        /** Pointer to weights */
        std::shared_ptr<base::Array<T>> wgt;

        /** Indicate if linear layer (alternate fashion window buffer) */
        bool linear = false;

        /** Indicate if RNN layer (different dimensions) */
        bool _3dim = false;

        /** Stride of the layer */
        int stride = 0;

        /** Number of effective concurrent multiplications per PE */
        uint32_t EF_LANES = 0;

        /** Number of effective columns */
        uint32_t EF_COLUMNS = 0;

        /** Number of efffective rows */
        uint32_t EF_ROWS = 0;

        /** Number of physical columns per window */
        uint32_t ACT_BLKS = 0;

        /** Number of physical rows per filter */
        uint32_t WGT_BLKS = 0;

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
         * @param _abuffer
         * @param _wbuffer
         * @param _obuffer
         */
        Control(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_wbuffer, const std::shared_ptr<LocalBuffer<T>> &_obuffer) :
                scheduler(_scheduler), dram(_dram), gbuffer(_gbuffer), abuffer(_abuffer), wbuffer(_wbuffer),
                obuffer(_obuffer) {}

        uint32_t getActBlks() const;

        uint32_t getWgtBlks() const;

        const std::shared_ptr<DRAM<T>> &getDram() const;

        const std::shared_ptr<GlobalBuffer<T>> &getGbuffer() const;

        const std::shared_ptr<LocalBuffer<T>> &getAbuffer() const;

        const std::shared_ptr<LocalBuffer<T>> &getWbuffer() const;

        const std::shared_ptr<LocalBuffer<T>> &getObuffer() const;

        const std::shared_ptr<Architecture<T>> &getArch() const;

        void setArch(const std::shared_ptr<Architecture<T>> &_arch);

        /**
        * Return name for the dataflow
        * @return Name
        */
        virtual std::string dataflow() = 0;

        /**
         * Configure control values for the current layer
         */
        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
                bool __3dim, int _stride);

        const std::vector<AddressRange> &getReadActAddresses() const;

        const std::vector<AddressRange> &getReadWgtAddresses() const;

        bool getIfEvictAct() const;

        bool getIfEvictWgt() const;

        bool still_off_chip_data();

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        virtual bool still_on_chip_data(std::vector<TileData<T>> &tiles_data) = 0;

        bool check_if_write_output(std::vector<TileData<T>> &tiles_data);

    };

}

#endif //DNNSIM_CONTROL_H
