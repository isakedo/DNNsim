#ifndef DNNSIM_WINDOWFIRSTOUTS_H
#define DNNSIM_WINDOWFIRSTOUTS_H

#include "OutputStationary.h"

namespace core {

    /**
     * Window first output stationary dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class WindowFirstOutS : public OutputStationary<T> {

    private:

        /**
         * Return name for the dataflow
         * @return Name of the dataflow
         */
        std::string name() override;

        void generate_address_maps() override;

        void generate_execution_graph_conv_layer();

        void generate_execution_graph_linear_layer();

        void generate_execution_graph() override;

        void configure_layer(const std::shared_ptr<base::Array<T>> &_act, const std::shared_ptr<base::Array<T>> &_wgt,
                bool _diffy, bool _schedule, bool _fc, bool _lstm, int _recurrences, int _out_x, int _out_y,
                int _stride, uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES) override;
        /**
         * Return if still data to process for convolutional layers
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data_conv_layer(std::vector<TileData<T>> &tiles_data);

        /**
         * Return if still data to process for linear layers
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data_linear_layer(std::vector<TileData<T>> &tiles_data);

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data(std::vector<TileData<T>> &tiles_data) override;

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
        WindowFirstOutS(const BitTactical<T> &_scheduler, uint64_t _data_size, uint64_t _global_buffer_size,
                uint64_t _act_buffer_size, uint64_t _wgt_buffer_size, uint64_t _start_act_address,
                uint64_t _start_wgt_address) : OutputStationary<T>(_scheduler, _data_size, _global_buffer_size,
                _act_buffer_size, _wgt_buffer_size, _start_act_address, _start_wgt_address) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
