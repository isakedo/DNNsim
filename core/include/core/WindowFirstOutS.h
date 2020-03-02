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
        std::string dataflow() override;

        void generate_execution_graph_conv_layer();

        void generate_execution_graph_linear_layer();

        void generate_execution_graph() override;

        void configure_layer(const std::shared_ptr<base::Array<T>> &_act, const std::shared_ptr<base::Array<T>> &_wgt,
                bool _linear, bool _lstm, int _stride, uint32_t _EF_COLUMNS, uint32_t _EF_ROWS) override;
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

        WindowFirstOutS(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<Architecture<T>> &_arch) :
                OutputStationary<T>(_scheduler, _dram, _gbuffer, _arch) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
