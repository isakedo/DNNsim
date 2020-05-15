#ifndef DNNSIM_CONTROL_H
#define DNNSIM_CONTROL_H

#include "Utils.h"
#include "BitTactical.h"

#include "DRAM.h"
#include "GlobalBuffer.h"
#include "LocalBuffer.h"

#include "Composer.h"
#include "PPU.h"

#include "Architecture.h"

namespace core {

    /**
     * Control Logic
     * @tparam T Data type values
     */
    template <typename T>
    class Control {

    protected:

        /** On-chip stage memory node */
        class Node {
        public:

            /** True if evict previous activations from on-chip */
            bool evict_act = false;

            /** True if evict previous weights from on-chip */
            bool evict_wgt = false;

            /** True if activations already on-chip */
            bool layer_act_on_chip = false;

            /** Activation addresses to read */
            std::vector<AddressRange> read_act_addresses;

            /** Weight addresses to read */
            std::vector<AddressRange> read_wgt_addresses;

            /** Output activation addresses to write */
            std::vector<AddressRange> write_addresses;
        };

        /** List of on-chip stages */
        std::vector<std::shared_ptr<Node>> on_chip_graph;

        /** Weight buffer scheduler */
        std::shared_ptr<BitTactical<T>> scheduler;

        /** Dram model */
        std::shared_ptr<DRAM<T>> dram;

        /** Global Buffer model */
        std::shared_ptr<GlobalBuffer<T>> gbuffer;

        /** Activation Buffer model */
        std::shared_ptr<LocalBuffer<T>> abuffer;

        /** Weight Buffer model */
        std::shared_ptr<LocalBuffer<T>> wbuffer;

        /** Output Buffer model */
        std::shared_ptr<LocalBuffer<T>> obuffer;

        /** Composer Column model */
        std::shared_ptr<Composer<T>> composer;

        /** Post-Processing Unit model */
        std::shared_ptr<PPU<T>> ppu;

        /** Architecture processing engine model */
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

        /** Number of effective rows */
        uint32_t EF_ROWS = 0;

        /** Number of physical columns per window */
        uint32_t ACT_BLKS = 0;

        /** Number of physical rows per filter */
        uint32_t WGT_BLKS = 0;

        /** True if activations already on-chip */
        bool layer_act_on_chip = false;

        /** True if activations on-chip for the next layer */
        bool next_layer_act_on_chip = false;

        /** Generate memory mapping for activations */
        virtual void generate_memory_maps() = 0;

        /** Generate execution graph */
        virtual void generate_execution_graph() = 0;

    public:

        /**
         * Constructor
         * @param _scheduler    Weight buffer scheduler
         * @param _dram         Dram model
         * @param _gbuffer      Global Buffer model
         * @param _abuffer      Activation Buffer model
         * @param _wbuffer      Weight Buffer model
         * @param _obuffer      Output Buffer model
         * @param _composer     Composer column model
         * @param _ppu          Post-Processing Unit model
         */
        Control(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_wbuffer, const std::shared_ptr<LocalBuffer<T>> &_obuffer,
                const std::shared_ptr<Composer<T>> &_composer, const std::shared_ptr<PPU<T>> &_ppu) :
                scheduler(_scheduler), dram(_dram), gbuffer(_gbuffer), abuffer(_abuffer), wbuffer(_wbuffer),
                obuffer(_obuffer), composer(_composer), ppu(_ppu) {}

        /**
         * Return a pointer to the dram model
         * @return Dram model
         */
        const std::shared_ptr<DRAM<T>> &getDram() const;

        /**
         * Return a pointer to the global buffer model
         * @return Global Buffer model
         */
        const std::shared_ptr<GlobalBuffer<T>> &getGbuffer() const;

        /**
         * Return a pointer to the Activation Buffer model
         * @return Activation Buffer model
         */
        const std::shared_ptr<LocalBuffer<T>> &getAbuffer() const;

        /**
         * Return a pointer to the Weight Buffer model
         * @return Weight Buffer model
         */
        const std::shared_ptr<LocalBuffer<T>> &getWbuffer() const;

        /**
         * Return a pointer to the Output Buffer model
         * @return Output Buffer model
         */
        const std::shared_ptr<LocalBuffer<T>> &getObuffer() const;

        /**
         * Return a pointer to the Composer Column model
         * @return Composer Column model
         */
        const std::shared_ptr<Composer<T>> &getComposer() const;

        /**
         * Return a pointer to the Post-Processing Unit model
         * @return Post-Processing Unit model
         */
        const std::shared_ptr<PPU<T>> &getPPU() const;

        /**
         * Return a pointer to the Architecture model
         * @return Architecture model
         */
        const std::shared_ptr<Architecture<T>> &getArch() const;

        /**
         * Update the architecture pointer
         * @param _arch Architecture model
         */
        void setArch(const std::shared_ptr<Architecture<T>> &_arch);

        /**
        * Return name for the dataflow
        * @return Name Dataflow name
        */
        virtual std::string dataflow() = 0;

        /**
         * Configure control values for the current layer
         * @param _act      Pointer to activation values
         * @param _wgt      Pointer to weight values
         * @param act_prec  Activations precision
         * @param wgt_prec  Weight precision
         * @param _linear   True if linear layer
         * @param __3dim    True if layer has 3 dimensions
         * @param _stride   Stride
         */
        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
                bool __3dim, int _stride);

        /**
         * Return activation addresses to read for the current node
         * @return Activation addresses to read
         */
        const std::vector<AddressRange> &getReadActAddresses() const;

        /**
         * Return weight addresses to read for the current node
         * @return Weight addresses to read
         */
        const std::vector<AddressRange> &getReadWgtAddresses() const;

        /**
         * Return output activation addresses to write for the current node
         * @return Output Activation addresses to write
         */
        const std::vector<AddressRange> &getWriteAddresses() const;

        /**
         * Return True if evict previous activations from on-chip for the current node
         * @return Evict Activation
         */
        bool getIfEvictAct() const;

        /**
         * Return True if evict previous weights from on-chip for the current node
         * @return Evict Weight
         */
        bool getIfEvictWgt() const;

        /**
         * Update the memory node and return if more on-chip stages
         * @return True if still off-chip data to process
         */
        bool still_off_chip_data();

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        virtual bool still_on_chip_data(std::vector<TileData<T>> &tiles_data) = 0;

        /**
         * Return true if there is output values to write to the global buffer
         * @param tiles_data Current data to process in the tiles
         * @return True if data to write
         */
        bool check_if_write_output(std::vector<TileData<T>> &tiles_data);

    };

}

#endif //DNNSIM_CONTROL_H
