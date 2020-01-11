#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include <sys/common.h>
#include <sys/Stats.h>
#include <sys/Batch.h>

#include <base/Array.h>
#include <base/Layer.h>
#include <base/Network.h>
#include <interface/NetReader.h>

#include "Dataflow.h"
#include "Architecture.h"
#include "BitTactical.h"
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
    class Simulator {

    private:

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

    public:

        /** Constructor
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
        Simulator(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES, uint32_t _BITS_PE,
                uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : N_LANES(_N_LANES),
                N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES), BITS_PE(_BITS_PE), N_THREADS(_N_THREADS),
                FAST_MODE(_FAST_MODE),  QUIET(_QUIET), CHECK(_CHECK) {}

        /** Simulate architecture for the given network
        * @param network   Network we want to calculate work reduction
        * @param arch      Pointer to the architecture to simulate
        * @param dataflow  Pointer to the dataflow
        */
        void run(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch,
                 const std::shared_ptr<Dataflow<T>> &dataflow);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         * @param arch      Pointer to the architecture to simulate
         */
        void potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch);

    };

}

#endif //DNNSIM_SIMULATOR_H
