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

        /** List of X coordinates for the windows */
        std::vector<int> x_windows;

        /** List of X coordinates for the windows */
        std::vector<int> y_windows;

        /** Coord X windows counter */
        int x_counter = 0;

        /** Coord Y windows counter */
        int y_counter = 0;

        /** Skip variable for bit tactical */
        int skip = 0;

        /**
         * Return name for the dataflow
         * @return Name
         */
        std::string name();

        /**
         *
         * @param _act
         * @param _wgt
         * @param _schedule
         * @param _N_LANES
         * @param _N_COLUMNS
         * @param _N_ROWS
         * @param _N_TILES
         */
        void initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, uint32_t _N_LANES, uint32_t _N_COLUMNS,
                uint32_t _N_ROWS, uint32_t _N_TILES);

        /**
         *
         * @param _batch
         */
        void initialise_batch(int _batch);

        /**
         * Return if schedule the weight buffer
         * @param act_row
         * @param wgt_row
         * @param _x_windows
         * @param _y_windows
         * @param _set
         * @return True if weight buffer to schedule, False if not
         */
        bool next_dataflow_step(BufferRow<T> &act_row, BufferRow<T> &wgt_row, std::vector<int> &_x_windows,
                std::vector<int> &_y_windows, int &_set);

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit WindowFirstOutS(const BitTactical<T> &_scheduler) : Dataflow<T>(_scheduler) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
