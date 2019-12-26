#ifndef DNNSIM_DATAFLOW_H
#define DNNSIM_DATAFLOW_H

#include "Utils.h"
#include "BitTactical.h"

namespace core {

    /**
     * Generic Dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class Dataflow {

    protected:

        /** Pointer to activations */
        std::shared_ptr<base::Array<T>> act;

        /** Pointer to weights */
        std::shared_ptr<base::Array<T>> wgt;

        /** Weight buffer scheduler */
        BitTactical<T> scheduler;

        /** Schedule weight buffer */
        bool schedule = false;

        /** Indicate if FC layer (alternate fashion window buffer) */
        bool fc = false;

        /** Indicate if LSTM layer (different dimensions order) */
        bool lstm = false;

        /** Current batch for the dataflow */
        int batch = 0;

        /** Number of recurrences for Recurrent neural network */
        int recurrence = 0;

        /** Output window X dimensions */
        int out_x = 0;

        /** Output window Y dimensions */
        int out_y = 0;

        /** Stride of the layer */
        int stride = 0;

        /** Number of concurrent multiplications per PE */
        uint32_t N_LANES;

        /** Number of columns */
        uint32_t N_COLUMNS;

        /** Number of rows */
        uint32_t N_ROWS;

        /** Number of tiles */
        uint32_t N_TILES;

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit Dataflow(const BitTactical<T> &_scheduler) : scheduler(_scheduler), N_LANES(0), N_COLUMNS(0),
                N_ROWS(0), N_TILES(0) {}

        /**
        * Return name for the dataflow
        * @return Name
        */
        virtual std::string name() = 0;

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
        virtual void initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, bool _fc, bool _lstm, int _recurrence,
                int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS,
                uint32_t _N_TILES) {
            act = _act;
            wgt = _wgt;
            schedule = _schedule;
            fc = _fc;
            lstm = _lstm;
            recurrence = _recurrence;
            out_x = _out_x;
            out_y = _out_y;
            stride = _stride;
            N_LANES = _N_LANES;
            N_COLUMNS = _N_COLUMNS;
            N_ROWS = _N_ROWS;
            N_TILES = _N_TILES;
        }

        /**
         *
         * @param _batch
         */
        virtual void initialise_batch(int _batch) {
            batch = _batch;
        }

        /**
         *
         * @param tile_data
         * @return
         */
        virtual bool next_dataflow_step(std::vector<TileData<T>> &tile_data) = 0;

    };

}

#endif //DNNSIM_DATAFLOW_H
