#ifndef DNNSIM_WINDOWFIRSTOUTS_H
#define DNNSIM_WINDOWFIRSTOUTS_H

#include "Dataflow.h"

namespace core {

    /**
     * Window first output stationary dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class WindowFirstOutS : public Dataflow<T> {

    private:

        /** Weight buffer */
        Buffer<T> weight_buffer;

        /** Window buffer */
        BufferSet<T> window_buffer;

        /** Number of window sets */
        uint64_t window_sets = 0;

        /** Number of filter sets */
        uint64_t filter_sets = 0;

        /** Maximum buffer depth */
        uint64_t max_buffer_time = 0;

        /** List of coordinates for the windows */
        std::vector<WindowCoord> windows;

        /** List of filters per tile */
        std::vector<std::vector<int>> filters;

        /** Recurrence counter */
        int current_recurrence = 0;

        /** Window counter */
        int window_set = 0;

        /** Filter counter */
        int filter_set = 0;

        /** Time counter */
        int time = 0;

        /** Skip variable for bit tactical */
        int skip = 0;

        /** Indicate if window buffer already filled */
        bool window_buffer_filled = false;

        /** Indicate if filter buffer already filled */
        bool filter_buffer_filled = false;

        /**
         * Return name for the dataflow
         * @return Name
         */
        std::string name() override;

        /**
         *
         * @param _act
         * @param _wgt
         * @param _schedule
         * @param _fc
         * @param _lstm
         * @param _recurrence
         * @param _out_x
         * @param _out_y
         * @param _stride
         * @param _N_LANES
         * @param _N_COLUMNS
         * @param _N_ROWS
         * @param _N_TILES
         */
        void initialise_layer(const std::shared_ptr<base::Array<T>> &_act, const std::shared_ptr<base::Array<T>> &_wgt,
                bool _schedule, bool _fc, bool _lstm, int _recurrence, int _out_x, int _out_y, int _stride,
                uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES) override;

        /**
         *
         * @param _batch
         */
        void initialise_batch(int _batch) override;

        /**
         * Return if schedule the weight buffer
         * @param tile_data
         * @return True if weight buffer to schedule, False if not
         */
        bool next_dataflow_step(std::vector<TileData<T>> &tile_data) override;

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit WindowFirstOutS(const BitTactical<T> &_scheduler) : Dataflow<T>(_scheduler) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
