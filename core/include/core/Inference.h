#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include <sys/common.h>
#include <sys/Stats.h>
#include <sys/Batch.h>

#include <base/Array.h>
#include <base/Layer.h>
#include <base/Network.h>
#include <interface/NetReader.h>

#include "Architecture.h"
#include "BitTactical.h"
#include "Memory.h"
#include "Utils.h"

#ifdef OPENMP
#include <omp.h>
#endif

namespace core {

    /**
     * Base class simulator for inference
     * @tparam T Data type of the simulation
     */
    template <typename T>
    class Inference {

    private:

        /** Memory abstraction of the simulator */
        Memory memory;

        /** Weight buffer scheduler */
        BitTactical<T> scheduler;

        /** Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /** Number of columns */
        const uint32_t N_COLUMNS;

        /** Number of rows */
        const uint32_t N_ROWS;

        /** Number of tiles */
        const uint32_t N_TILES;

        /** Bits per PE */
        const uint32_t BITS_PE;

        /** Number of parallel cores */
        const int N_THREADS;

        /** Enable fast mode: only one image */
        const bool FAST_MODE = false;

        /** Avoid std::out messages */
        const bool QUIET = false;

        /** Check the correctness of the simulations */
        const bool CHECK = false;

        /**
         *
         * @param output
         * @param window_buffer
         * @param weight_buffer
         * @param x_windows
         * @param y_windows
         * @param num_filters
         * @param set
         */
        void calculate_output(OutputTensor &output, const BufferRow<ValueTuple<T>> &window_buffer,
                const BufferRow<ValueTuple<T>> &weight_buffer, const std::vector<int> &x_windows,
                const std::vector<int> &y_windows, uint64_t num_filters, int set);

        /**
         * Fill the weight buffer with the weights
         * @param weight_buffer Empty weight buffer (Overwritten)
         * @param wgt           Weight array
         * @param num_filters   Number of filters
         * @param wgt_channels  Number of weight channels
         * @param Kx            Kernel width
         * @param Ky            Kernel height
         */
        void fill_weight_buffer(Buffer<ValueTuple<T>> &weight_buffer, const base::Array<T> &wgt, uint64_t num_filters,
                uint64_t wgt_channels, uint64_t Kx, uint64_t Ky);

        /**
         *
         * @param window_buffer
         * @param act
         * @param x_windows
         * @param y_windows
         * @param n
         * @param act_channels
         * @param Kx
         * @param Ky
         * @param stride
         */
        void fill_window_buffer(BufferSet<ValueTuple<T>> &window_buffer, const base::Array<T> &act,
                const std::vector<int> &x_windows, const std::vector<int> &y_windows, uint64_t n, uint64_t act_channels,
                uint64_t Kx, uint64_t Ky, int stride);

    public:

        /** Constructor
         * @param _scheduler    Weight buffer scheduler
         * @param _N_LANES      Number of concurrent multiplications per PE
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _N_TILES      Number of tiles
         * @param _BITS_PE      Number of bits per PE
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        Inference(const BitTactical<T> &_scheduler, uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS,
                uint32_t _N_TILES, uint32_t _BITS_PE, uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) :
                scheduler(_scheduler), memory(Memory()), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS),
                N_TILES(_N_TILES), BITS_PE(_BITS_PE), N_THREADS(_N_THREADS), FAST_MODE(_FAST_MODE),  QUIET(_QUIET),
                CHECK(_CHECK) {}

        /** Simulate architecture for the given network
        * @param network   Network we want to calculate work reduction
        * @param arch      Pointer to the architecture to simulate
        */
        void run(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         * @param arch      Pointer to the architecture to simulate
         */
        void potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch);

    };

}

#endif //DNNSIM_SIMULATOR_H
