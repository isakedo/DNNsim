#ifndef DNNSIM_CONTROL_H
#define DNNSIM_CONTROL_H

#include "Utils.h"
#include "BitTactical.h"

namespace core {

    /**
     * Control Logic
     * @tparam T Data type values
     */
    template <typename T>
    class Control {

    protected:

        /** Weight buffer scheduler */
        BitTactical<T> scheduler;

        uint64_t data_size;

        uint64_t global_buffer_size;

        uint64_t act_buffer_size;

        uint64_t wgt_buffer_size;

        uint64_t start_act_address;

        uint64_t start_wgt_address;

        uint64_t global_buffer_banks;

        uint64_t global_buffer_bank_width;

        /** Pointer to activations */
        std::shared_ptr<base::Array<T>> act;

        /** Pointer to weights */
        std::shared_ptr<base::Array<T>> wgt;

        /** Schedule weight buffer */
        bool schedule = false;

        /** Diffy simulation */
        bool diffy = false;

        /** Indicate if FC layer (alternate fashion window buffer) */
        bool fc = false;

        /** Indicate if LSTM layer (different dimensions order) */
        bool lstm = false;

        /** Number of recurrences for Recurrent neural network */
        int recurrences = 0;

        /** Output window X dimensions */
        int out_x = 0;

        /** Output window Y dimensions */
        int out_y = 0;

        /** Stride of the layer */
        int stride = 0;

        /** Number of concurrent multiplications per PE */
        uint32_t N_LANES = 0;

        /** Number of columns */
        uint32_t N_COLUMNS = 0;

        /** Number of rows */
        uint32_t N_ROWS = 0;

        /** Number of tiles */
        uint32_t N_TILES = 0;

        virtual void generate_memory_maps() = 0;

        virtual void generate_execution_graph() = 0;

    public:

        /**
         * Constructor
         * @param _scheduler
         * @param _data_size
         * @param _global_buffer_size
         * @param _act_buffer_size
         * @param _wgt_buffer_size
         * @param _global_buffer_banks
         * @param _global_buffer_bank_width
         * @param _start_act_address
         * @param _start_wgt_address
         */
        Control(const BitTactical<T> &_scheduler, uint64_t _data_size, uint64_t _global_buffer_size,
                uint64_t _act_buffer_size, uint64_t _wgt_buffer_size, uint64_t _global_buffer_banks,
                uint64_t _global_buffer_bank_width, uint64_t _start_act_address, uint64_t _start_wgt_address) :
                scheduler(_scheduler),  data_size(_data_size), global_buffer_size(_global_buffer_size),
                act_buffer_size(_act_buffer_size), wgt_buffer_size(_wgt_buffer_size),
                global_buffer_banks(_global_buffer_banks), global_buffer_bank_width(_global_buffer_bank_width),
                start_act_address(_start_act_address), start_wgt_address(_start_wgt_address) {}

        /**
        * Return name for the dataflow
        * @return Name
        */
        virtual std::string name() = 0;

        /**
         * Configure control values for the current layer
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
                uint32_t _N_ROWS, uint32_t _N_TILES) {
            act = _act;
            wgt = _wgt;
            diffy = _diffy;
            schedule = _schedule;
            fc = _fc;
            lstm = _lstm;
            recurrences = _recurrences;
            out_x = _out_x;
            out_y = _out_y;
            stride = _stride;
            N_LANES = _N_LANES;
            N_COLUMNS = _N_COLUMNS;
            N_ROWS = _N_ROWS;
            N_TILES = _N_TILES;
        }

        virtual bool still_off_chip_data() = 0;

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        virtual bool still_on_chip_data(std::vector<TileData<T>> &tiles_data) = 0;

    };

}

#endif //DNNSIM_CONTROL_H
