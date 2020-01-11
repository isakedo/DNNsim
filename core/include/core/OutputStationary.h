#ifndef DNNSIM_OUTPUTSTATIONARY_H
#define DNNSIM_OUTPUTSTATIONARY_H

#include "Dataflow.h"

namespace core {

    /**
     * Generic Output Stationary dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class OutputStationary : public Dataflow<T> {

    protected:

        /** Weight buffer */
        Buffer<T> weight_buffer;

        /** Window buffer */
        BufferSet<T> window_buffer;

        /** Number of window sets */
        uint64_t window_sets = 0;

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
        int current_recurrence = 0;

        /** Group counter */
        int group = 0;

        /** Window counter */
        int window_set = 0;

        /** Filter counter */
        int filter_set = 0;

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
         *
         */
        void fill_window_buffer();

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit OutputStationary(const BitTactical<T> &_scheduler) : Dataflow<T>(_scheduler) {}

        /**
        * Return name for the dataflow
        * @return Name
        */
        virtual std::string name() = 0;

        /**
         * Initialise values for the current layer
         * @param _act          Activation array
         * @param _wgt          Weight array
         * @param _diffy        Diffy
         * @param _schedule     Schedule buffer
         * @param _fc           Fully connected
         * @param _lstm         LSTM
         * @param _recurrence   Recurrences
         * @param _out_x        Output X windows
         * @param _out_y        Output Y windows
         * @param _stride       Stride
         * @param _N_LANES      Number of lanes
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _N_TILES      Number of tiles
         */
        void initialise_layer(const std::shared_ptr<base::Array<T>> &_act, const std::shared_ptr<base::Array<T>> &_wgt,
                bool _diffy, bool _schedule, bool _fc, bool _lstm, int _recurrence, int _out_x, int _out_y, int _stride,
                uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES) override;

        /**
         * Return if schedule the weight buffer
         * @param tile_data
         * @return True if weight buffer to schedule, False if not
         */
        virtual bool next_dataflow_step(std::vector<TileData<T>> &tile_data) = 0;

    };

}

#endif //DNNSIM_OUTPUTSTATIONARY_H
