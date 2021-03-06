#ifndef DNNSIM_OUTPUTSTATIONARY_H
#define DNNSIM_OUTPUTSTATIONARY_H

#include "Control.h"

namespace core {

    enum MemPolicy { ALL, INPUTS, SET, SUBSET, CHANNELS, GROUPS };

    /**
     * Generic Output Stationary dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class OutputStationary : public Control<T> {

    protected:

        /** On-chip stage memory node for Output Stationary dataflow */
        class NodeOutS : public Control<T>::Node {
        public:

            /** Current recurrence step in the RNN layer */
            uint64_t recurrence = 0;

            /** Current time step when computing the output in different memory stages */
            uint64_t time_step = 0;

            /** Maximum buffer time */
            uint64_t max_time = 0;

            /** Group ID to process */
            std::vector<int> groups;

            /** Window ID to process */
            std::vector<int> window_sets;

            /** Filter ID to process */
            std::vector<int> filter_sets;

            /** Use the previous activation buffer from previous node */
            bool use_prev_buffer = false;
        };

        /** Weight buffer */
        Buffer<T> weight_buffer;

        /** Weight Addresses buffer */
        AddressBuffer wgt_address_buffer;

        /** Weight Addresses map */
        std::vector<AddressRange> wgt_address_map;

        /** Weight banks buffer */
        BankBuffer wgt_bank_buffer;

        /** Weight End time */
        std::vector<uint64_t> wgt_end_time;

        /** Window buffer */
        BufferSet<T> window_buffer;

        /** Window Addresses buffer */
        AddressBufferSet window_address_buffer;

        /** Activation Addresses map */
        AddressMap act_address_map;

        /** Window Bank buffer */
        BankBufferSet window_bank_buffer;

        /** Activation Bank map */
        ActBankMap act_bank_map;

        /** Pointer to the next activation address in the mapping */
        uint64_t next_act_address = 0;

        /** Pointer to the next weight address in the mapping */
        uint64_t next_wgt_address = 0;

        /** Pointer to the next output address in the mapping */
        uint64_t next_out_address = 0;

        /** Output width size */
        int out_x = 0;

        /** Output height size */
        int out_y = 0;

        /** Depthwise layer flag */
        bool depthwise = false;

        /** Number of layer groups */
        uint32_t groups = 0;

        /** Number of window sets */
        uint32_t window_sets = 0;

        /** Number of filter sets */
        uint32_t filter_sets = 0;

        /** Number of filters per group */
        uint32_t filters_per_group = 0;

        /** Maximum buffer depth */
        uint32_t max_buffer_time = 0;

        /** List of coordinates for the windows */
        std::vector<WindowCoord> windows;

        /** List of filters per tile */
        std::vector<std::vector<int>> filters;

        /** Group iterator */
        int group_it = 0;

        /** Window iterator */
        int window_set_it = 0;

        /** Filter iterator */
        int filter_set_it = 0;

        /** Last requested time */
        int requested = 0;

        /** Write */
        std::vector<bool> write;

        /** Time counter */
        std::vector<int> time;

        /** Skip variable for bit tactical */
        std::vector<int> skip;

        /** Indicate if window buffer already filled */
        bool window_buffer_filled = false;

        /** Indicate if filter buffer already filled */
        bool filter_buffer_filled = false;

        /** Track if the tiles are done */
        bool tiles_done = false;

        /**
         * Schedule how many input windows can fit on-chip per memory step
         * @param window_steps Window indices split by on-chip steps
         * @param output_size Output size per window
         * @param channels Total number of channels per window to schedule
         */
        void fill_window_steps(std::vector<std::vector<int>> &window_steps, uint32_t output_size, uint32_t channels);

        /**
         * Generate the addresses to read from off-chip given the window limits
         * @param start_act_blk     Start activation memory block
         * @param end_act_blk       End activation memory block
         * @param last_act_blk      Last memory block to read
         * @param start_window      First window to read
         * @param end_window        Last window to read
         * @param start_group       Start group for grouping layers
         * @return                  Addresses to read from off-chip
         */
        std::vector<AddressRange> generate_addresses(uint32_t start_act_blk, uint32_t end_act_blk,
                uint32_t last_act_blk, uint32_t start_window, uint32_t end_window, uint32_t start_group);

        /**
         * Generate memory mapping for input data
         */
        void generate_memory_maps() override;

        /**
         * Fill the weight buffer with the weights
         */
        void fill_weight_buffer();

        /**
         * Fill the window buffer with the activations to process
         */
        void fill_window_buffer(uint32_t group_idx);

        /**
         * Return the number of outputs in the current node step
         * @return Outputs on-chip
         */
        uint64_t calculate_outputs() override;

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

    public:

        /**
         * Constructor
         * @param _scheduler    Weight buffer scheduler
         * @param _dram         Dram model
         * @param _gbuffer      Global Buffer model
         * @param _abuffer      Activation Buffer model
         * @param _pbuffer      Weight Buffer model
         * @param _wbuffer      Partial Sum Buffer model
         * @param _obuffer      Output Buffer model
         * @param _composer     Composer column model
         * @param _ppu          Post-Processing Unit model
         */
        OutputStationary(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_pbuffer, const std::shared_ptr<LocalBuffer<T>> &_wbuffer,
                const std::shared_ptr<LocalBuffer<T>> &_obuffer, const std::shared_ptr<Composer<T>> &_composer,
                const std::shared_ptr<PPU<T>> &_ppu) : Control<T>(_scheduler,_dram, _gbuffer, _abuffer, _pbuffer,
                _wbuffer, _obuffer, _composer, _ppu) {}

    };

}

#endif //DNNSIM_OUTPUTSTATIONARY_H
