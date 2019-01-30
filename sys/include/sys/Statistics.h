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

            /* Stats for cycles */
            std::vector<std::vector<uint32_t>> cycles;
            std::vector<uint32_t> avg_cycles;
            std::vector<uint32_t> baseline_cycles;

            /* SCNN */
            std::vector<std::vector<uint32_t>> dense_cycles;
            std::vector<std::vector<uint32_t>> mults;
            std::vector<std::vector<uint32_t>> idle_bricks;
            std::vector<std::vector<uint32_t>> idle_conflicts;
            std::vector<std::vector<uint32_t>> idle_pe;
            std::vector<std::vector<uint32_t>> idle_halo;
            std::vector<std::vector<uint32_t>> halo_transfers;
            std::vector<std::vector<uint32_t>> weight_buff_reads;
            std::vector<std::vector<uint32_t>> act_buff_reads;
            std::vector<std::vector<uint32_t>> accumulator_updates;
            std::vector<std::vector<uint32_t>> i_loop;
            std::vector<std::vector<uint32_t>> f_loop;
            std::vector<std::vector<uint32_t>> offchip_weight_reads;

            std::vector<uint32_t> avg_dense_cycles;
            std::vector<uint32_t> avg_mults;
            std::vector<uint32_t> avg_idle_bricks;
            std::vector<uint32_t> avg_idle_conflicts;
            std::vector<uint32_t> avg_idle_pe;
            std::vector<uint32_t> avg_idle_halo;
            std::vector<uint32_t> avg_halo_transfers;
            std::vector<uint32_t> avg_weight_buff_reads;
            std::vector<uint32_t> avg_act_buff_reads;
            std::vector<uint32_t> avg_accumulator_updates;
            std::vector<uint32_t> avg_i_loop;
            std::vector<uint32_t> avg_f_loop;
            std::vector<uint32_t> avg_offchip_weight_reads;

            /* Stats for potentials */
            std::vector<std::vector<double>> work_reduction;
            std::vector<double> avg_work_reduction;
            std::vector<std::vector<double>> speedup;
            std::vector<double> avg_speedup;
            std::vector<std::vector<uint64_t>> bit_multiplications;
            std::vector<uint64_t> avg_bit_multiplications;
            std::vector<uint64_t> parallel_multiplications;

            template <typename T>
            T get_average(const std::vector<T> &vector_stat) const {
                return accumulate(vector_stat.begin(), vector_stat.end(), 0.0) / vector_stat.size();
            }

            template <typename T>
            T get_total(const std::vector<T> &vector_stat) const {
                return accumulate(vector_stat.begin(), vector_stat.end(), 0.0);
            }

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
