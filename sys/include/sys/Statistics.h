#ifndef DNNSIM_STATISTICS_H
#define DNNSIM_STATISTICS_H

#include <sys/common.h>

namespace sys {

    class Statistics {

    public:

        /* Make public to be easier to operate */
        struct Stats {

            /* Name of the task done */
            std::string task_name;

            /* Name of the network */
            std::string net_name;

            /* Simulator architecture */
            std::string arch;

            /* Layer vector */
            std::vector<std::string> layers;

            /* Activations precision */
            std::vector<int> act_prec;

            /* Weights precision */
            std::vector<int> wgt_prec;

            /* Computation time per layer */
            std::vector<std::chrono::duration<double>> time;

            /* Stats for Pragmatic cycles */
            std::vector<std::vector<uint32_t>> PRA_cycles;
            std::vector<uint32_t> PRA_avg_cycles;
            std::vector<uint32_t> PRA_baseline_cycles;

            /* Stats for Laconic cycles */
            std::vector<std::vector<uint32_t>>  LAC_cycles;
            std::vector<uint32_t> LAC_avg_cycles;

            /* Stats for potentials */
            std::vector<std::vector<double>> potentials;
            std::vector<double> avg_potentials;
            std::vector<std::vector<uint64_t>> bit_multiplications;
            std::vector<uint64_t> avg_bit_multiplications;
            std::vector<uint64_t> parallel_multiplications;

            /* Stats for memory accesses */
            std::vector<uint32_t> on_chip_weights;
            std::vector<uint32_t> on_chip_activations;
            std::vector<uint32_t> off_chip_weights_sch3;
            std::vector<uint32_t> bits_weights;
            std::vector<uint32_t> off_chip_weights_sch4;
            std::vector<uint32_t> bits_working_weights;
            std::vector<uint32_t> off_chip_activations;
            std::vector<uint32_t> bits_one_activation_row;
            std::vector<uint32_t> computations;

        };

    private:

        /* Set of statistics containing all simulation stats. Shared across all system */
        static std::vector<Stats> all_stats;

    public:

        /* Initializes values of the struct
         * @param stats Stats struct we want to initialize
         */
        static void initialize(Stats &stats);

        /* Add one stats struct to the set of statistics
         * @param _stats    Stats struct that is going to be added
         */
        static void addStats(const Stats &_stats);

        /* Getter */
        static const std::vector<Stats> &getAll_stats();

    };

}

#endif //DNNSIM_STATISTICS_H
