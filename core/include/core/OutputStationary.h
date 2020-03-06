#ifndef DNNSIM_OUTPUTSTATIONARY_H
#define DNNSIM_OUTPUTSTATIONARY_H

#include "Control.h"

namespace core {

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
            uint64_t start_channel = 0;
            uint64_t end_channel = 0;
            std::vector<int> groups;
            std::vector<int> window_sets;
            std::vector<int> filter_sets;
        };

        struct BufferAddresses {
            uint64_t first_address = 0;
            uint64_t last_address = 0;
        };

        /** Weight buffer */
        Buffer<T> weight_buffer;

        /** Weight Addresses buffer */
        AddressBuffer wgt_address_buffer;

        /** Weight Addresses map */
        BufferAddresses wgt_address_map;

        /** Weight banks buffer */
        BankBuffer wgt_bank_buffer;

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

        int out_x = 0;

        int out_y = 0;

        /** Number of layer groups */
        uint64_t groups = 0;

        /** Number of window sets */
        uint64_t window_sets = 0;

        /** Number of filter sets */
        uint64_t filter_sets = 0;

        /** Number of fitlers per group */
        uint64_t filters_per_group = 0;

        /** Maximum buffer depth */
        uint64_t max_buffer_time = 0;

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

        /** Time counter */
        std::vector<int> time;

        /** Skip variable for bit tactical */
        std::vector<int> skip;

        /** Indicate if window buffer already filled */
        bool window_buffer_filled = false;

        /** Indicate if filter buffer already filled */
        bool filter_buffer_filled = false;

        void generate_memory_maps() override;

        /**
         * Fill the weight buffer with the weights
         */
        void fill_weight_buffer();

        /**
         * Fill the window buffer with the activations to process
         */
        void fill_window_buffer(uint64_t group_idx);

        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _linear, bool _lstm, int _stride,
                uint32_t _EF_COLUMNS, uint32_t _EF_ROWS);

    public:

        OutputStationary(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_wbuffer, const std::shared_ptr<LocalBuffer<T>> &_obuffer) :
                Control<T>(_scheduler,_dram, _gbuffer, _abuffer, _wbuffer, _obuffer) {}

    };

}

#endif //DNNSIM_OUTPUTSTATIONARY_H
