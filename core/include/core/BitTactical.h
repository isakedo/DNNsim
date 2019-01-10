#ifndef DNNSIM_BITTACTICAL_H
#define DNNSIM_BITTACTICAL_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitTactical : public Simulator<T> {

    private:

        /* Compute the potentials for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computeMemAccessesConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

    protected:

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Lookahead value of D*/
        const int LOOKAHEAD_D;

        /* Lookaside value of H*/
        const int LOOKASIDE_H;

        /* Search shape for the scheduler: must be 'L' or 'T' */
        const char SEARCH_SHAPE;

        /*
         *
         */
        std::vector<std::vector<std::queue<std::tuple<int,int,int>>>> scheduler(const cnpy::Array<T> &wgt,
                int act_channels);

        bool check_schedule(const std::vector<std::vector<std::queue<std::tuple<int,int,int>>>> &dense_schedule,
                             int init_filter, int max_filter);

        void update_schedule(std::vector<std::vector<std::queue<std::tuple<int,int,int>>>> &dense_schedule,
                int init_filter, int max_filter);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        virtual void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats) = 0;

        /* Compute the timing for a fully-connected layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        virtual void computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) = 0;

        /* Compute the potentials for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        virtual void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) = 0;

        /* Compute the potentials for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        virtual void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats) = 0;

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        virtual void run(const Network<T> &network) = 0;

        /* Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        virtual void potentials(const Network<T> &network) = 0;

        /* Constructor
         * @param _N_COLUMNS            Number of columns
         * @param _N_ROWS               Number of rows
         * @param _LOOKAHEAD_D          Value for scheduler lookahead
         * @param _LOOKASIDE_H          Value for scheduler lookaside
         * @param _SEARCH_SHAPE         Type of search
         */
        BitTactical(int _N_COLUMNS, int _N_ROWS, int _LOOKAHEAD_D, int _LOOKASIDE_H, const char _SEARCH_SHAPE) :
            N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), LOOKAHEAD_D(_LOOKAHEAD_D), LOOKASIDE_H(_LOOKASIDE_H),
            SEARCH_SHAPE(_SEARCH_SHAPE) {}

    public:

        /* Calculate the number of memory accesses
         * @param network   Network we want to simulate
         */
        void memoryAccesses(const Network<T> &network);

    };

}

#endif //DNNSIM_BITTACTICAL_H
