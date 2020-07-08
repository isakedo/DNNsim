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

        /**
         * Generate execution graph for convolutional layers
         */
        void generate_execution_graph_conv_layer();

        /**
         * Generate execution graph for grouped layers
         */
        void generate_execution_graph_grouped_layer();

        /**
         * Generate execution graph for linear layers
         */
        void generate_execution_graph_linear_layer();

        /**
         * Generate execution graph
         */
        void generate_execution_graph() override;

        /**
         * Configure control values for the current layer
         * @param _act      Pointer to activation values
         * @param _wgt      Pointer to weight values
         * @param act_prec  Activations precision
         * @param wgt_prec  Weight precision
         * @param _linear   True if linear layer
         * @param __3dim    True if layer has 3 dimensions
         * @param _stride   Stride
         */
        void configure_layer(const std::shared_ptr<base::Array<T>> &_act, const std::shared_ptr<base::Array<T>> &_wgt,
                uint32_t act_prec, uint32_t wgt_prec, bool _linear, bool __3dim, int _stride) override;
        /**
         * Return if still data to process for convolutional layers
         * @param _tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data_conv_layer(TilesData<T> &_tiles_data);

        /**
         * Return if still data to process for linear layers
         * @param _tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data_linear_layer(TilesData<T> &_tiles_data);

        /**
         * Return if still data to process
         * @param tiles_data Tile data to process
         * @return True if still data to process, False if not
         */
        bool still_on_chip_data(TilesData<T> &tiles_data) override;

    public:

        /**
         * Constructor
         * @param _scheduler    Weight buffer scheduler
         * @param _dram         Dram model
         * @param _gbuffer      Global Buffer model
         * @param _abuffer      Activation Buffer model
         * @param _pbuffer      Weight Buffer model
         * @param _wbuffer      Partial Sum Buffer model
         * @param _obuffer      Output Buffer model
         * @param _composer     Composer column model
         * @param _ppu          Post-Processing Unit model
         */
        WindowFirstOutS(const std::shared_ptr<BitTactical<T>> &_scheduler, const std::shared_ptr<DRAM<T>> &_dram,
                const std::shared_ptr<GlobalBuffer<T>> &_gbuffer, const std::shared_ptr<LocalBuffer<T>> &_abuffer,
                const std::shared_ptr<LocalBuffer<T>> &_pbuffer, const std::shared_ptr<LocalBuffer<T>> &_wbuffer,
                const std::shared_ptr<LocalBuffer<T>> &_obuffer, const std::shared_ptr<Composer<T>> &_composer,
                const std::shared_ptr<PPU<T>> &_ppu) : OutputStationary<T>(_scheduler, _dram, _gbuffer, _abuffer,
                _pbuffer, _wbuffer, _obuffer, _composer, _ppu) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
