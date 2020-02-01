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

        struct NodeOutS {
            uint64_t recurrence = 0;
            uint64_t start_channel = 0;
            uint64_t end_channel = 0;
            std::vector<int> window_sets;
            std::vector<int> filter_tile_sets;
            std::vector<AddressRange> read_addresses;
            std::vector<AddressRange> clean_addresses;
            std::vector<AddressRange> write_addresses;
        };

        std::vector<NodeOutS> on_chip_graph;

        /** Weight buffer */
        Buffer<T> weight_buffer;

        /** Window buffer */
        BufferSet<T> window_buffer;

        /** Number of filter sets */
        uint64_t filter_sets = 0;

        /** Number of groups */
        uint64_t groups = 0;

        /** Maximum buffer depth */
        uint64_t max_buffer_time = 0;

        /** Filters per group */
        uint64_t filters_per_group = 0;

        /** Window buffer depth for all groups */
        uint64_t max_window_buffer_time = 0;

        /** List of coordinates for the windows */
        std::vector<WindowCoord> windows;

        /** List of filters per tile */
        std::vector<std::vector<int>> filters;

        /** Recurrence counter */
        //int current_recurrence = 0;

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

        /**
         * Fill the weight buffer with the weights
         */
        void fill_weight_buffer();

        /**
         * Fill the window buffer with the activations to process
         */
        void fill_window_buffer();

        /**
         * Initialise values for the current layer
         * @param _act          Activation array
         * @param _wgt          Weight array
         * @param _diffy        Diffy
         * @param _schedule     Schedule buffer
         * @param _fc           Fully connected
         * @param _lstm         LSTM
         * @param _recurrences  Recurrences
         * @param _out_x        Output X windows
         * @param _out_y        Output Y windows
         * @param _stride       Stride
         * @param _N_LANES      Number of lanes
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _N_TILES      Number of tiles
         */
        virtual void configure_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _diffy, bool _schedule, bool _fc, bool _lstm,
                int _recurrences, int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS,
                uint32_t _N_ROWS, uint32_t _N_TILES);

        bool still_off_chip_data() override;

    public:

        /**
         * Constructor
         * @param _scheduler
         * @param _data_size
         * @param _global_buffer_size
         * @param _act_buffer_size
         * @param _wgt_buffer_size
         * @param _start_act_address
         * @param _start_wgt_address
         */
        OutputStationary(const BitTactical<T> &_scheduler, uint64_t _data_size, uint64_t _global_buffer_size,
                uint64_t _act_buffer_size, uint64_t _wgt_buffer_size, uint64_t _start_act_address,
                uint64_t _start_wgt_address) : Control<T>(_scheduler, _data_size, _global_buffer_size, _act_buffer_size,
                _wgt_buffer_size, _start_act_address, _start_wgt_address) {}

    };

}

#endif //DNNSIM_OUTPUTSTATIONARY_H
