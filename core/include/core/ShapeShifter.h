#ifndef DNNSIM_SHAPESHIFTER_H
#define DNNSIM_SHAPESHIFTER_H

#include "Architecture.h"

namespace core {

    /**
     * ShapeShifter simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class ShapeShifter : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Number of activations per group */
        const uint32_t PRECISION_GRANULARITY;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Calculate also the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /** Diffy simulation */
        const bool DIFFY;

        /** BitTactical simulation */
        const bool TCT;

        /* AUXILIARY FUNCTIONS */

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
         * Return if schedule the weight buffer
         * @return True if weight buffer to schedule, False if not
         */
        bool schedule() override;

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
         * @param act_prec      Activation layer precision
         * @param wgt_prec      Weight layer precision
         * @param network_bits  Maximum number of bits in the network
         * @return              Number of one bit multiplications
         */
        uint16_t computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) override;

    public:

        /** Constructor
         * @param _PRECISION_GRANULARITY    Granularity for dynamic precisions
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _MINOR_BIT                Calculate also the minor bit for dynamic precisions
         * @param _DIFFY                    Enable Diffy
         * @param _TCT                      Enable BitTactical simulation
         */
        ShapeShifter(uint32_t _PRECISION_GRANULARITY, uint32_t _COLUMN_REGISTERS, bool _MINOR_BIT, bool _DIFFY,
                bool _TCT) : PRECISION_GRANULARITY(_PRECISION_GRANULARITY), COLUMN_REGISTERS(_COLUMN_REGISTERS),
                MINOR_BIT(_MINOR_BIT), DIFFY(_DIFFY), TCT(_TCT) {}

    };

}

#endif //DNNSIM_SHAPESHIFTER_H
