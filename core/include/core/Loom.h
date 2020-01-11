#ifndef DNNSIM_LOOM_H
#define DNNSIM_LOOM_H

#include "Architecture.h"

namespace core {

    /**
     * Loom simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class Loom : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Number of columns/rows per group */
        const uint32_t GROUP_SIZE;

        /** Number of bits in series that the PE process */
        const uint32_t PE_SERIAL_BITS;

        /** Calculate also the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /** Calculate dynamic precision for weights rather than profiled */
        const bool DYNAMIC_WEIGHTS;

        /** Activations mask to remove negative numbers */
        uint16_t act_mask = 0;

        /** Weights mask to remove negative numbers */
        uint16_t wgt_mask = 0;

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec     Activations precision
         * @param _wgt_prec     Weights precision
         * @param _network_bits Network bits
         * @param _linear       Linear layer
         * @param COLUMNS       Number of columns
         * @param TILES         Number of tiles
         */
        void initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear, uint64_t COLUMNS,
                uint64_t TILES) override;

        /**
         * Get number of cycles
         * @return Cycles
         */
        uint64_t getCycles() const override;

        /**
         * Return name of the class
         * @return Name
         */
        std::string name() override;

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec) override;

        /* CYCLES */

        /**
         * Return stats filename for the architecture in the cycles function
         * @return Filename
         */
        std::string filename() override;

        /**
         * Return stats header for the architecture in the cycles function
         * @return Header
         */
        std::string header() override;

        /**
         * Return if calculate deltas for the window buffer
         * @return True if diffy, False if not
         */
        bool diffy() override;

        /**
         * Return if schedule the weight buffer
         * @return True if weight buffer to schedule, False if not
         */
        bool schedule() override;

        /**
         * Calculate cycles for linear layers
         * @param tile_data Processing information for all the tiles
         */
        void process_linear(const std::vector<TileData<T>> &tiles_data);

        /**
         * Calculate cycles for convolutional layers
         * @param tile_data Processing information for all the tiles
         */
        void process_convolution(const std::vector<TileData<T>> &tiles_data);

        /**
         * Calculate cycles for all the tiles
         * @param tiles_data Processing information for all the tiles
         */
        void process_tiles(const std::vector<TileData<T>> &tiles_data) override;

        /* POTENTIALS */

        /**
         * Return stats filename for the architecture in the potentials function
         * @return Filename
         */
        std::string filename_pot() override;

        /**
         * Return stats header for the architecture in the potentials function
         * @return Header
         */
        std::string header_pot() override;

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @return              Number of one bit multiplications
         */
        uint16_t computeBits(T act, T wgt) override;

    public:

        /** Constructor
         * @param _GROUP_SIZE               Granularity for dynamic precisions
         * @param _PE_SERIAL_BITS           Number of bits in series that the PE process
         * @param _MINOR_BIT                Calculate also the minor bit for dynamic precisions
         * @param _DYNAMIC_WEIGHTS          Calculate dynamic precision for weights rather than profiled
         */
        Loom(uint32_t _GROUP_SIZE, uint32_t _PE_SERIAL_BITS, bool _MINOR_BIT, bool _DYNAMIC_WEIGHTS) :
                GROUP_SIZE(_GROUP_SIZE), PE_SERIAL_BITS(_PE_SERIAL_BITS), MINOR_BIT(_MINOR_BIT),
                DYNAMIC_WEIGHTS(_DYNAMIC_WEIGHTS) {}

    };

}

#endif //DNNSIM_LOOM_H
