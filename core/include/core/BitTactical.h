#ifndef DNNSIM_BITTACTICAL_H
#define DNNSIM_BITTACTICAL_H

#include "Utils.h"

namespace core {

    /**
     * BitTactical scheduler
     * @tparam T 16 bits fixed point or 32 bits floating-point
     */
    template <typename T>
    class BitTactical {

    private:

        /** Lookahead value of H*/
        const uint32_t LOOKAHEAD_H;

        /** Lookaside value of D*/
        const uint32_t LOOKASIDE_D;

        /** Search shape for the scheduler: must be 'L' or 'T' */
        const char SEARCH_SHAPE;

        /** Number of concurrent multiplications per PE */
        uint32_t N_LANES;

        /** Search space for the scheduler */
        std::vector<std::tuple<int, int>> SEARCH_MAP;

    public:

        /** Constructor
         * @param _LOOKAHEAD_H      Value for scheduler lookahead
         * @param _LOOKASIDE_D      Value for scheduler lookaside
         * @param _SEARCH_SHAPE     Type of search
         */
        BitTactical(uint32_t _LOOKAHEAD_H, uint32_t _LOOKASIDE_D, const char _SEARCH_SHAPE) : LOOKAHEAD_H(_LOOKAHEAD_H),
                LOOKASIDE_D(_LOOKASIDE_D), SEARCH_SHAPE(_SEARCH_SHAPE), N_LANES(0) {

            if (SEARCH_SHAPE == 'L') {

                for (int h = 1; h <= LOOKAHEAD_H; ++h)
                    SEARCH_MAP.emplace_back(std::make_tuple(h,0));

                for (int d = 1; d <= LOOKASIDE_D; ++d)
                    SEARCH_MAP.emplace_back(std::make_tuple(1,-d));

            } else if (SEARCH_SHAPE == 'T') {

                for (int h = 1; h <= LOOKAHEAD_H; ++h)
                    SEARCH_MAP.emplace_back(std::make_tuple(h,0));

                int h = 1;
                int d = 1;
                bool sign = false;
                for (int i = 0; i < LOOKASIDE_D; ++i) {
                    SEARCH_MAP.emplace_back(std::make_tuple(h,d));
                    d *= -1;
                    if (sign) {
                        d++;
                        h++;
                        if (h > LOOKAHEAD_H) h = 1;
                        sign = false;
                    } else
                        sign = true;
                }

            }

            std::sort(SEARCH_MAP.begin(), SEARCH_MAP.end());

        }

        /**
         * Return lookahead value
         * @return Lookahead
         */
        uint32_t getLookaheadH() const;

        /**
         * Check the whole rows is zeroes
         * @param buffer Schedule buffer row
         * @return True if all row is zero
         */
        bool check_zero_line(const BufferRow<T> &buffer);

        /**
         * Promote one effectual candidate to the ineffectual value position
         * @param buffer Schedule buffer (Overwritten)
         * @param ineffectual Ineffectual value (zero value)
         * @param candidate Effectual value to promote (non-zero value)
         */
        void promote(BufferSet<T> &buffer, ValueIndex ineffectual, ValueIndex candidate);

        /**
         * Search effectual values in the search space
         * @param buffer Schedule buffer
         * @param value_idx Time and lane from which to search
         * @param max_time Maximum time that can be promoted
         * @return List of indices for the candidate effectual values
         */
        std::vector<ValueIndex> search(const BufferSet<T> &buffer, ValueIndex value_idx, int max_time);

        /**
         * Schedule buffer set using original schedule
         * @param buffer Buffer set to scheduler (Overwritten)
         */
        void original_schedule(BufferSet<T> &buffer);

        /**
         * Schedule buffer
         * @param buffer Buffer to scheduler (Overwritten)
         * @param _N_LANES Number of lanes
         */
        void schedule(Buffer<T> &buffer, uint32_t _N_LANES);

    };

}

#endif //DNNSIM_BITTACTICAL_H
