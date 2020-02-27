#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Architecture.h"

namespace core {

    /**
     * Laconic simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class Laconic : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Activate booth-like encoding */
        const bool BOOTH_ENCODING;

        /* AUXILIARY FUNCTIONS */

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
         * @param tiles_data Processing information for all the tiles
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

        /**
         * Return true if ready to feed need data
         * @return True if ready to process data
         */
        bool ready() override;

        /**
         * Return true if processing has finished
         * @return True if done
         */
        bool flush() override;

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
         * @param _N_LANES          Number of concurrent multiplications per PE
         * @param _N_COLUMNS        Number of columns
         * @param _N_ROWS           Number of rows
         * @param _N_TILES          Number of tiles
         * @param _BITS_PE          Bits per PE
         * @param _BOOTH_ENCODING   Activate booth-like encoding
         */
        Laconic(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES, uint32_t _BITS_PE,
                bool _BOOTH_ENCODING) : Architecture<T>(_N_LANES, _N_COLUMNS, _N_ROWS, _N_TILES, _BITS_PE),
                BOOTH_ENCODING(_BOOTH_ENCODING) {}

    };

}

#endif //DNNSIM_LACONIC_H
