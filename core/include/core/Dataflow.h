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
        bool schedule;

        /** Current batch for the dataflow */
        int batch;

        /** Number of concurrent multiplications per PE */
        uint32_t N_LANES;

        /** Number of columns */
        uint32_t N_COLUMNS;

        /** Number of rows */
        uint32_t N_ROWS;

        /** Number of tiles */
        uint32_t N_TILES;

        /**
         * Fill the weight buffer with the weights
         * @param weight_buffer Empty weight buffer (Overwritten)
         * @param wgt           Weight array
         * @param num_filters   Number of filters
         * @param wgt_channels  Number of weight channels
         * @param Kx            Kernel width
         * @param Ky            Kernel height
         */
        void fill_weight_buffer(Buffer<T> &weight_buffer, uint64_t num_filters, uint64_t wgt_channels, uint64_t Kx,
                uint64_t Ky);

        /**
         *
         * @param window_buffer
         * @param act
         * @param x_windows
         * @param y_windows
         * @param n
         * @param act_channels
         * @param Kx
         * @param Ky
         * @param stride
         */
        void fill_window_buffer(BufferSet<T> &window_buffer, const std::vector<int> &x_windows,
                const std::vector<int> &y_windows, uint64_t n, uint64_t act_channels, uint64_t Kx, uint64_t Ky,
                int stride);

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit Dataflow(const BitTactical<T> &_scheduler) : scheduler(_scheduler), schedule(false), batch(0),
                N_LANES(0), N_COLUMNS(0), N_ROWS(0), N_TILES(0) {}

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
         * @param _N_LANES
         * @param _N_COLUMNS
         * @param _N_ROWS
         * @param _N_TILES
         */
        virtual void initialise_layer(const std::shared_ptr<base::Array<T>> &_act,
                const std::shared_ptr<base::Array<T>> &_wgt, bool _schedule, uint32_t _N_LANES, uint32_t _N_COLUMNS,
                uint32_t _N_ROWS, uint32_t _N_TILES);

        /**
         *
         * @param _batch
         */
        virtual void initialise_batch(int _batch);

        /**
         * Return if schedule the weight buffer
         * @param act_row
         * @param wgt_row
         * @param _x_windows
         * @param _y_windows
         * @param _set
         * @return True if weight buffer to schedule, False if not
         */
        virtual bool next_dataflow_step(BufferRow<T> &act_row, BufferRow<T> &wgt_row, std::vector<int> &_x_windows,
                std::vector<int> &_y_windows, int &_set) = 0;

    };

}

#endif //DNNSIM_DATAFLOW_H
