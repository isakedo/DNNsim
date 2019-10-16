#ifndef DNNSIM_DYNAMICTACTICAL_H
#define DNNSIM_DYNAMICTACTICAL_H

#include "Simulator.h"

typedef std::tuple<uint16_t, uint16_t> value_index;
typedef std::tuple<float, uint16_t, uint16_t> value_mux;
typedef std::vector<std::vector<std::vector<float>>> non_schedule_buffer;
typedef std::vector<std::vector<value_mux>> schedule_buffer;

namespace core {

    /**
     * DynamicTactical simulator
     * @tparam T 16 bits bfloat
     */
    template <typename T>
    class DynamicTactical : public Simulator<T> {

    private:

        /** Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /** Number of columns */
        const uint32_t N_COLUMNS;

        /** Number of rows */
        const uint32_t N_ROWS;

        /** Number of tiles */
        const uint32_t N_TILES;

        /** Lookahead value of H*/
        const uint32_t LOOKAHEAD_H;

        /** Lookaside value of D*/
        const uint32_t LOOKASIDE_D;

        /** Search shape for the scheduler: must be 'L' or 'T' */
        const char SEARCH_SHAPE;

        /** Search space for the scheduler */
        std::vector<std::tuple<int8_t, int8_t>> SEARCH_MAP;

        /** Compute number of one bit multiplications given a first and a second value
         * @param first         First value
         * @param second        Second value
         * @param first_value   if True check sparsity in first value, if not in second
         * @return              Number of one bit multiplications
         */
        uint16_t computeDynamicTacticalBitsPE(T first, T second, bool first_value);

        /**
         * Promote one effectual candidate to the ineffectual value position
         * @param schedule Schedule buffer (Overwritten)
         * @param ineffectual Ineffectual value (zero value)
         * @param candidate Effectual value to promote (non-zero value)
         */
        void promote(schedule_buffer &schedule, value_index ineffectual, value_index candidate);

        /**
         * Search effectual values in the search space
         * @param schedule Schedule buffer
         * @param value_idx Time and lane from which to search
         * @param max_time Maximum time that can be promoted
         * @return List of indices for the candidate effectual values
         */
        std::vector<value_index> search(const schedule_buffer &schedule, value_index value_idx, int max_time);

        /**
         * Schedule buffer using original schedule
         * @param schedule Buffer to scheduler (Overwritten)
         */
        void original_schedule(schedule_buffer &schedule);

    public:

        /** Constructor
         * @param _N_LANES          Number of concurrent multiplications per PE
         * @param _N_COLUMNS        Number of columns
         * @param _N_ROWS           Number of rows
         * @param _N_TILES          Number of tiles
         * @param _LOOKAHEAD_H      Value for scheduler lookahead
         * @param _LOOKASIDE_D      Value for scheduler lookaside
         * @param _SEARCH_SHAPE     Type of search
         * @param _N_THREADS        Number of parallel threads for multi-threading execution
         * @param _FAST_MODE        Enable fast mode to simulate only one image
         * @param _QUIET            Avoid std::out messages
         * @param _CHECK            Check the correctness of the simulations
         */
        DynamicTactical(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES,
                uint32_t _LOOKAHEAD_H, uint32_t _LOOKASIDE_D, const char _SEARCH_SHAPE, uint8_t _N_THREADS,
                bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,_QUIET,_CHECK),
                N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES), LOOKAHEAD_H(_LOOKAHEAD_H),
                LOOKASIDE_D(_LOOKASIDE_D), SEARCH_SHAPE(_SEARCH_SHAPE) {

            if (SEARCH_SHAPE == 'L') {

                for (int h = 1; h <= LOOKAHEAD_H; ++h)
                    SEARCH_MAP.emplace_back(std::make_tuple(h,0));

                for (int d = 1; d <= LOOKASIDE_D; ++d)
                    SEARCH_MAP.emplace_back(std::make_tuple(1,-d));

            } else if (SEARCH_SHAPE == 'T') {

                for (int h = 1; h <= LOOKAHEAD_H; ++h)
                    SEARCH_MAP.emplace_back(std::make_tuple(h,0));

                for (int d = 1; d <= LOOKAHEAD_H; ++d)
                    SEARCH_MAP.emplace_back(std::make_tuple(d,-d));

                for (int d = 1; d <= LOOKAHEAD_H; ++d)
                    SEARCH_MAP.emplace_back(std::make_tuple(d,d));

                SEARCH_MAP.emplace_back(std::make_tuple(1,-3));

            }
        }

        /** Run the timing simulator of the architecture
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         */
        void run(const sys::Batch::Simulate &simulate, int epochs);

        /** Calculate work reduction for the given network
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         */
        void potentials(const sys::Batch::Simulate &simulate, int epochs);

    };

}

#endif //DNNSIM_BITTACTICAL_H
