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

    protected:

        /* SIMULATION PARAMETERS */

        /** Column index */
        uint64_t column_index = 0;

        /** Column cycles */
        std::vector<uint64_t> column_cycles;

        /** Activations precision */
        int act_prec = 0;

        /** Weights precision */
        int wgt_prec = 0;

        /** Network bits */
        int network_bits = 0;

        /** Linear layer */
        bool linear = false;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

        /** Ready cycle */
        uint64_t ready_cycle = 0;

        /** Done cycle */
        uint64_t done_cycle = 0;

        /* STATISTICS */

        /** Compute cycles */
        std::vector<uint64_t> compute_cycles;

        /** Number of cycles */
        uint64_t cycles = 0;

        /** Number of stalle cycles */
        uint64_t stall_cycles = 0;

        /** Scheduled PEs */
        uint64_t scheduled_pe = 0;

        /** Idle PEs */
        uint64_t idle_pe = 0;

    public:

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec     Activations precision
         * @param _wgt_prec     Weights precision
         * @param _network_bits Network bits
         * @param _linear       Linear layer
         * @param COLUMNS       Number of columns
         */
        virtual void initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear, uint64_t COLUMNS) {
            act_prec = _act_prec;
            wgt_prec = _wgt_prec;
            network_bits = _network_bits;
            linear = _linear;
            ready_cycle = 0;
            done_cycle = 0;

            column_cycles = std::vector<uint64_t>(COLUMNS, 0);
            column_index = 0;

            compute_cycles = std::vector<uint64_t>(COLUMNS, 0);
            cycles = 0;
            stall_cycles = 0;
            scheduled_pe = 0;
            idle_pe = 0;
        }

        /**
         * Set shared global cycle
         * @param globalCycle
         */
        void setGlobalCycle(const std::shared_ptr<uint64_t> &globalCycle) {
            global_cycle = globalCycle;
        }

        /**
         * Get number of cycles
         * @return Cycles
         */
        virtual uint64_t getCycles() const {
            return cycles;
        }

        /**
         * Get number of stall cycles
         * @return Sstall cycles
         */
        uint64_t getStallCycles() const {
            return stall_cycles;
        }

        /**
         * Get scheduled PEs
         * @return Scheduled PEs
         */
        uint64_t getScheduledPe() const {
            return scheduled_pe;
        }

        /**
         * Get idle PEs
         * @return Idle PEs
         */
        uint64_t getIdlePe() const {
            return idle_pe;
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
         * Return if calculate deltas for the window buffer
         * @return True if diffy, False if not
         */
        virtual bool diffy() = 0;

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

        /**
         * Return true if ready to feed need data
         * @return True if ready to process data
         */
        virtual bool ready() { return ready_cycle <= *global_cycle; }

        /**
         * Return true if processing is done
         * @return True if done
         */
        virtual bool done() { return done_cycle <= *global_cycle; }

        /**
         * Return true if processing has finished
         * @return True if done
         */
        virtual bool flush() { return done_cycle <= *global_cycle; }

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
         * @param act   Activation
         * @param wgt   Weight
         * @return      Number of one bit multiplications
         */
        virtual uint16_t computeBits(T act, T wgt) = 0;

    };

}

#endif //DNNSIM_ARCHITECTURE_H
