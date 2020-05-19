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

        class NodeOutS : public Control<T>::Node {
        public:
            uint64_t recurrence = 0;
            uint64_t time_step = 0;
            uint64_t max_time = 0;
            std::vector<int> groups;
            std::vector<int> window_sets;
            std::vector<int> filter_sets;
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

        uint64_t next_act_address = 0;

        uint64_t next_wgt_address = 0;

        uint64_t next_out_address = 0;

        int out_x = 0;

        int out_y = 0;

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

        /** Write */
        std::vector<bool> write;

        /** Time counter */
        std::vector<int> time;

        /** Skip variable for bit tactical */
        std::vector<int> skip;

        int prev_filter_set = 0;

        /** Indicate if window buffer already filled */
        bool window_buffer_filled = false;

        /** Indicate if filter buffer already filled */
        bool filter_buffer_filled = false;

        void fill_window_steps(std::vector<std::vector<int>> &window_steps, uint32_t output_size, uint32_t channels);

        std::vector<AddressRange> generate_addresses(uint32_t start_act_blk, uint32_t end_act_blk,
                uint32_t last_act_blk, uint32_t start_window, uint32_t end_window, uint32_t start_group);

        void generate_memory_maps() override;

        /**
         * Fill the weight buffer with the weights
         */
        void fill_weight_buffer();

        /**
         * Fill the window buffer with the activations to process
         */
        void fill_window_buffer(uint32_t group_idx);

        uint64_t calculate_outputs() override;

        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
                bool __3dim, int _stride);

    public:

        OutputStationary(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_wbuffer, const std::shared_ptr<LocalBuffer<T>> &_obuffer,
                const std::shared_ptr<Composer<T>> &_composer, const std::shared_ptr<PPU<T>> &_ppu) :
                Control<T>(_scheduler,_dram, _gbuffer, _abuffer, _wbuffer, _obuffer, _composer, _ppu) {}

    };

}

#endif //DNNSIM_OUTPUTSTATIONARY_H
