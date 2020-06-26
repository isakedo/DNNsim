#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include <sys/common.h>
#include <sys/Stats.h>
#include <sys/Batch.h>

#include <base/Array.h>
#include <base/Layer.h>
#include <base/Network.h>

#include "Control.h"
#include "Architecture.h"
#include "DRAM.h"
#include "GlobalBuffer.h"
#include "BitTactical.h"
#include "Utils.h"

namespace core {

    enum Stage {
        FETCH,
        MEMORY_I,
        MEMORY_II,
        EXECUTION,
        WRITEBACK_I,
        WRITEBACK_II,
        Last = WRITEBACK_II
    };

    template <typename T>
    class Pipeline {
    public:

        std::vector<std::shared_ptr<TilesData<T>>> pipeline;

        explicit Pipeline(uint64_t _stages) {
            pipeline = std::vector<std::shared_ptr<TilesData<T>>>(_stages, std::shared_ptr<TilesData<T>>());
        }

        void fetch_data(const TilesData<T> &tiles_data) {
            pipeline.front() = std::make_shared<TilesData<T>>(tiles_data);
        }

        void end_stage(Stage stage) {
            pipeline[stage] = nullptr;
        }

        void move_stage(Stage stage) {
            pipeline[stage + 1] = pipeline[stage];
            pipeline[stage] = nullptr;
        }

        const std::shared_ptr<TilesData<T>> &getData(Stage stage) {
            return pipeline[stage];
        }

        bool isEmpty() {
            for (const auto &stage : pipeline)
                if (stage != nullptr)
                    return false;
            return true;
        }

        bool isFree(Stage stage) {
            return pipeline[stage] == nullptr;
        }

        bool isValid(Stage stage) {
            return pipeline[stage] != nullptr;
        }


    };

    /**
     * Base class simulator for inference
     * @tparam T Data type of the simulation
     */
    template <typename T>
    class Simulator {

    private:

        /** Enable fast mode: only one sample */
        const bool FAST_MODE = false;

        /** Avoid std::out messages */
        const bool QUIET = false;

        /** Check the correctness of the simulations */
        const bool CHECK = false;

    public:

        /** Constructor
         * @param _FAST_MODE    Enable fast mode to simulate only one sample
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        Simulator(bool _FAST_MODE, bool _QUIET, bool _CHECK) : FAST_MODE(_FAST_MODE), QUIET(_QUIET), CHECK(_CHECK) {}

        /** Simulate architecture for the given network
        * @param network   Network we want to calculate work reduction
        * @param control
        */
        void run(const base::Network<T> &network, const std::shared_ptr<Control<T>> &control);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         * @param arch      Pointer to the architecture to simulate
         */
        void potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch);

    };

}

#endif //DNNSIM_SIMULATOR_H
