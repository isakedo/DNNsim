#ifndef DNNSIM_BITTACTICAL_H
#define DNNSIM_BITTACTICAL_H

#include "Simulator.h"

namespace core {

    /**
     * BitTactical simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class BitTactical : public Simulator<T> {

    private:

        /** Search effectual weights in the defined search space
         * @param dense_schedule     Filter scheduled so far
         * @param wgt_index          Index of the ineffectual weight that is going to be substituted
         * @param max_time          Maximum time than can be scheduled (assuming stationary PSUM FIX)
         * @return                   Effectual candidates to substitute the ineffectual position
         */
        weights_set weight_search(const set_schedule &dense_schedule, weight_index wgt_idx, int max_time);

        /** Schedule the promotions for one filter given a specific time
         * @param dense_schedule    Schedule for a filter before removing zeroes (Overwritten)
         * @param time              Specific time to schedule
         * @param row               Row of X weight lanes to schedule
         * @param max_time          Maximum time than can be scheduled (assuming stationary PSUM FIX)
         */
        void filter_scheduler(set_schedule &dense_schedule, int time, int row, int max_time);

        /** Schedule the weights in the scratchpad removing zero weights
         * @param sparse_Schedule   Schedule of the weights without removing zeroes
         * @return                  Return the dense scheduled weights
         */
        schedule dense_scheduler(const schedule &sparse_schedule);

        /** Schedule the weights in the scratchpad without removing zero weights
         * @param wgt           Weights per layer
         * @param act_channels  Number of activation channels
         * @param fc            True if InnerProduct
         * @return              Return the sparse scheduled weights
         */
        schedule sparse_scheduler(const base::Array<T> &wgt, uint64_t act_channels, bool fc);

    protected:

        /** Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /** Number of columns */
        const uint32_t N_COLUMNS;

        /** Number of rows */
        const uint32_t N_ROWS;

        /** Number of rows */
        const uint32_t N_TILES;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Lookahead value of H*/
        const uint32_t LOOKAHEAD_H;

        /** Lookaside value of D*/
        const uint32_t LOOKASIDE_D;

        /** Search shape for the scheduler: must be 'L' or 'T' */
        const char SEARCH_SHAPE;

        /** Search space for the scheduler */
        std::vector<std::tuple<int, int>> SEARCH_MAP;

        /** Schedule the weights in the scratchpad trying to remove zero weights
         * @param wgt           Weights per layer
         * @param act_channels  Number of activation channels
         * @param fc            True if InnerProduct
         * @return              Return the scheduled weights
         */
        schedule scheduler(const base::Array<T> &wgt, uint64_t act_channels, bool fc);

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         * @param schedules Dense schedules for the layer we want to simulate
         */
        virtual void run(const base::Network<T> &network, const std::vector<schedule> &schedules) = 0;

        /** Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        virtual void potentials(const base::Network<T> &network) = 0;

        /** Constructor
         * @param _N_LANES          Number of concurrent multiplications per PE
         * @param _N_COLUMNS        Number of columns
         * @param _N_ROWS           Number of rows
         * @param _N_TILES          Number of tiles
         * @param _COLUMN_REGISTERS Number of registers per SIP
         * @param _LOOKAHEAD_H      Value for scheduler lookahead
         * @param _LOOKASIDE_D      Value for scheduler lookaside
         * @param _SEARCH_SHAPE     Type of search
         * @param _N_THREADS        Number of parallel threads for multi-threading execution
         * @param _FAST_MODE        Enable fast mode to simulate only one image
         * @param _QUIET            Avoid std::out messages
         * @param _CHECK            Check the correctness of the simulations
         */
        BitTactical(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES,
                uint32_t _COLUMN_REGISTERS, uint32_t _LOOKAHEAD_H, uint32_t _LOOKASIDE_D, const char _SEARCH_SHAPE,
                uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,
                _QUIET,_CHECK), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES),
                COLUMN_REGISTERS(_COLUMN_REGISTERS), LOOKAHEAD_H(_LOOKAHEAD_H), LOOKASIDE_D(_LOOKASIDE_D),
                SEARCH_SHAPE(_SEARCH_SHAPE) {

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

                SEARCH_MAP.emplace_back(std::make_tuple(1,LOOKAHEAD_H - 1));

            }
        }

    public:

        /** Return the weights scheduled for all the layers
         * @param network   Network we want to get the scheduler
         */
        std::vector<schedule> network_scheduler(const base::Network<T> &network);

    };

}

#endif //DNNSIM_BITTACTICAL_H
