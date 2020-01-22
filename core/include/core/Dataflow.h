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

        /** Diffy simulation */
        bool diffy = false;

        /** Indicate if FC layer (alternate fashion window buffer) */
        bool fc = false;

        /** Indicate if LSTM layer (different dimensions order) */
        bool lstm = false;

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

        /* STATISTICS */

        /** Activation buffer reads */
        uint64_t act_buff_reads = 0;

        /** Weight buffer reads */
        uint64_t wgt_buff_reads = 0;

        /** Accumulator updates */
        uint64_t acc_updates = 0;

        /** Output buffer writes */
        uint64_t out_buffer_writes = 0;

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit Dataflow(const BitTactical<T> &_scheduler) : scheduler(_scheduler), N_LANES(0), N_COLUMNS(0),
                N_ROWS(0), N_TILES(0) {}

        /**
         * Get activation buffer reads
         * @return Activation buffer reads
         */
        uint64_t getActBuffReads() const {
            return act_buff_reads;
        }

        /**
         * Get weight buffer reads
         * @return Weight buffer reads
         */
        uint64_t getWgtBuffReads() const {
            return wgt_buff_reads;
        }

        /**
         * Get Accumulator updates
         * @return Accumulator updates
         */
        uint64_t getAccUpdates() const {
            return acc_updates;
        }

        /**
         * Get output buffer writes
         * @return Output buffer writes
         */
        uint64_t getOutBufferWrites() const {
            return out_buffer_writes;
        }

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
        virtual void initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _diffy, bool _schedule, bool _fc, bool _lstm,
                int _recurrence, int _out_x, int _out_y, int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS,
                uint32_t _N_ROWS, uint32_t _N_TILES) {
            act = _act;
            wgt = _wgt;
            diffy = _diffy;
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

            act_buff_reads = 0;
            wgt_buff_reads = 0;
            acc_updates = 0;
            out_buffer_writes = 0;
        }

        /**
         * Return if still data to process
         * @param tile_data Tile data to process
         * @return True if still data to process, False if not
         */
        virtual bool next_dataflow_step(std::vector<TileData<T>> &tile_data) = 0;

    };

}

#endif //DNNSIM_DATAFLOW_H
