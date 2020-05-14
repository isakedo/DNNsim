#ifndef DNNSIM_ARCHITECTURE_H
#define DNNSIM_ARCHITECTURE_H

#include <sys/Stats.h>
#include "Utils.h"
#include <bits/stdc++.h>

namespace core {

    /**
     * Generic Architecture
     * @tparam T Data type values
     */
    template <typename T>
    class Architecture {

    protected:

        /** Number of concurrent multiplications per PE */
        const uint32_t LANES = 0;

        /** Number of columns */
        const uint32_t COLUMNS = 0;

        /** Number of rows */
        const uint32_t ROWS = 0;

        /** Number of tiles */
        const uint32_t TILES = 0;

        /** PE bit-width */
        const uint32_t PE_WIDTH = 0;

        /* SIMULATION PARAMETERS */

        /** Column index */
        uint64_t column_index = 0;

        /** Column cycles */
        std::vector<uint64_t> column_cycles;

        /** Activations precision */
        int act_prec = 0;

        /** Weights precision */
        int wgt_prec = 0;

        /** Network width */
        int network_width = 0;

        /** True if signed activations */
        bool signed_act = false;

        /** True if signed weights */
        bool signed_wgt = false;

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

        /**
         * Constructor
         */
        Architecture() = default;

        /**
         * Constructor
         * @param _LANES    Number of concurrent multiplications per PE
         * @param _COLUMNS  Number of columns
         * @param _ROWS     Number of rows
         * @param _TILES    Number of tiles
         * @param _PE_WIDTH    Bits per PE
         */
        Architecture(uint32_t _LANES, uint32_t _COLUMNS, uint32_t _ROWS, uint32_t _TILES, uint32_t _PE_WIDTH) :
                LANES(_LANES), COLUMNS(_COLUMNS), ROWS(_ROWS), TILES(_TILES), PE_WIDTH(_PE_WIDTH) {}

        /**
         * Return the number of lanes
         * @return Lanes
         */
        uint32_t getLanes() const {
            return LANES;
        }

        /**
         * Return the number of columns
         * @return Columns
         */
        uint32_t getColumns() const {
            return COLUMNS;
        }

        /**
         * Return the number of rows
         * @return Rows
         */
        uint32_t getRows() const {
            return ROWS;
        }

        /**
         * Return the number of tiles
         * @return Tiles
         */
        uint32_t getTiles() const {
            return TILES;
        }

        /**
         * Return the pe bit-width
         * @return PE width
         */
        uint32_t getPeWidth() const {
            return PE_WIDTH;
        }

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec      Activations precision
         * @param _wgt_prec      Weights precision
         * @param _network_width Network width
         * @param _signed_act    Signed activations
         * @param _signed_wgt    Signed weights
         * @param _linear        Linear layer
         * @param EF_COLUMNS     Number of effective columns
         */
        virtual void configure_layer(int _act_prec, int _wgt_prec, int _network_width, bool _signed_act,
                bool _signed_wgt, bool _linear, uint64_t EF_COLUMNS) {
            act_prec = _act_prec;
            wgt_prec = _wgt_prec;
            network_width = _network_width;
            signed_act = _signed_act;
            signed_wgt = _signed_wgt;
            linear = _linear;

            ready_cycle = 0;
            done_cycle = 0;

            column_cycles = std::vector<uint64_t>(EF_COLUMNS, 0);
            column_index = 0;

            compute_cycles = std::vector<uint64_t>(EF_COLUMNS, 0);
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
         * Return number of cycles
         * @return Cycles
         */
        virtual uint64_t getCycles() const {
            return cycles;
        }

        /**
         * Return number of stall cycles
         * @return Sstall cycles
         */
        uint64_t getStallCycles() const {
            return stall_cycles;
        }

        /**
         * Return scheduled PEs
         * @return Scheduled PEs
         */
        uint64_t getScheduledPe() const {
            return scheduled_pe;
        }

        /**
         * Return idle PEs
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
         */
        virtual void dataConversion(base::Array<T> &data) {}

        /* CYCLES */

        /**
         * Return stats filename for the architecture in the cycles function
         * @return Filename
         */
        virtual std::string filename() {
            return "_L" + std::to_string(LANES) + "_C" + std::to_string(COLUMNS) + "_R" + std::to_string(ROWS) +
                    "_T" + std::to_string(TILES) + "_BP" + std::to_string(PE_WIDTH);
        }

        /**
         * Return stats header for the architecture in the cycles function
         * @return Header
         */
        virtual std::string header() {
            std::string header = "Number of lanes/terms per PE: " + std::to_string(LANES) + "\n";
            header += "Number of columns/windows in parallel: " + std::to_string(COLUMNS) + "\n";
            header += "Number of rows/filters in parallel: " + std::to_string(ROWS) + "\n";
            header += "Number of tiles: " + std::to_string(TILES) + "\n";
            header += "PE input bit-width: " + std::to_string(PE_WIDTH) + "\n";
            return header;
        }

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
