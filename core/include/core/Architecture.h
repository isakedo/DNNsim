#ifndef DNNSIM_ARCHITECTURE_H
#define DNNSIM_ARCHITECTURE_H

#include <sys/Stats.h>
#include "Utils.h"

namespace core {

    /**
     * Generic Architecture
     * @tparam T Data type values
     */
    template <typename T>
    class Architecture {

    public:

        /* STATISTICS */

        /** Number of cycles */
        uint64_t cycles = 0;

        /** Number of stall cycles */
        uint64_t stall_cycles = 0;

        /** Scheduled PEs */
        uint64_t scheduled_pe = 0;

        /** Idle PEs */
        uint64_t idle_pe = 0;

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise stats to zero
         */
        void initialise_stats() {
            cycles = 0;
            stall_cycles = 0;
            scheduled_pe = 0;
        }

        /**
         * Return name of the class
         * @return Name
         */
        virtual std::string name() = 0;

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        virtual void dataConversion(base::Array<T> &data, uint8_t data_prec) = 0;

        /* CYCLES */

        /**
         * Return stats filename for the architecture in the cycles function
         * @return Filename
         */
        virtual std::string filename() = 0;

        /**
         * Return stats header for the architecture in the cycles function
         * @return Header
         */
        virtual std::string header() = 0;

        /**
         * Return if schedule the weight buffer
         * @return True if weight buffer to schedule, False if not
         */
        virtual bool schedule() = 0;

        /**
         * Calculate cycles for all the tiles
         * @param tiles_data Processing information for all the tiles
         */
        virtual void process_tiles(const std::vector<TileData<T>> &tiles_data) = 0;

        /* POTENTIALS */

        /**
         * Return stats filename for the architecture in the potentials function
         * @return Filename
         */
        virtual std::string filename_pot() = 0;

        /**
         * Return stats header for the architecture in the potentials function
         * @return Header
         */
        virtual std::string header_pot() = 0;

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param act_prec      Activation layer precision
         * @param wgt_prec      Weight layer precision
         * @param network_bits  Maximum number of bits in the network
         * @return              Number of one bit multiplications
         */
        virtual uint16_t computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) = 0;

    };

}

#endif //DNNSIM_ARCHITECTURE_H
