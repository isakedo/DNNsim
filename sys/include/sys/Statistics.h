#ifndef DNNSIM_STATISTICS_H
#define DNNSIM_STATISTICS_H

#include <sys/common.h>

namespace sys {

    class Statistics {

    public:

        /* Make public to be easier to operate */
        struct Stats {

            /* Simulation stats */
            std::string task_name;                                          //Name of the task done
            std::string net_name;                                           //Name of the network
            std::string arch;                                               //Simulator architecture
            bool tensorflow_8b;                                             //Tensforflow flag
            std::vector<std::string> layers;                                //Layers name
            std::vector<int> act_prec;                                      //Activations precision
            std::vector<int> wgt_prec;                                      //Weights precision

            /* Computation time per layer */
            std::vector<std::chrono::duration<double>> time;                //Execution time
            std::vector<std::vector<std::chrono::duration<double>>> training_time;  //Execution time for training traces

            /* Stats for cycles */
            std::vector<std::vector<uint64_t>> cycles;                      //Number of cycles
            std::vector<uint64_t> baseline_cycles;                          //Number of cycles of DaDianNao
            std::vector<std::vector<uint64_t>> weight_buff_reads;           //On-Chip Weight Buffer Reads
            std::vector<std::vector<uint64_t>> act_buff_reads;              //On-Chip Activation Buffer Reads
            std::vector<std::vector<uint64_t>> accumulator_updates;         //On-Chip Output Buffer Writes
            std::vector<std::vector<uint64_t>> scheduled_pe;                //Total scheduled Processing Engines
            std::vector<std::vector<uint64_t>> idle_pe;                     //Number of idle PEs in the whole simulation
            std::vector<std::vector<uint64_t>> stall_cycles;                //Column stall cyckes due to synchronization

            /* Stats for 8bits PEs */
            std::vector<uint64_t> idle_columns;                             //Idle columns due to spatial composition
            std::vector<uint64_t> idle_rows;                                //Idle rows due to spatial composition
            std::vector<uint64_t> columns_per_act;                          //Number of columns per activation window
            std::vector<uint64_t> rows_per_wgt;                             //Number of rows per weight filter

            /* SCNN */
            std::vector<std::vector<uint64_t>> dense_cycles;                //Number of cycles without zero skipping
            std::vector<std::vector<uint64_t>> mults;                       //Number of multiplications in total
            std::vector<std::vector<uint64_t>> idle_bricks;
            std::vector<std::vector<uint64_t>> idle_conflicts;
            std::vector<std::vector<uint64_t>> idle_column_cycles;
            std::vector<std::vector<uint64_t>> column_stalls;
            std::vector<std::vector<uint64_t>> idle_halo;
            std::vector<std::vector<uint64_t>> total_mult_cycles;
            std::vector<std::vector<uint64_t>> halo_transfers;
            std::vector<std::vector<uint64_t>> i_loop;
            std::vector<std::vector<uint64_t>> f_loop;
            std::vector<std::vector<uint64_t>> offchip_weight_reads;

            /* Bit Fusion */
            std::vector<uint64_t> perf_factor;
            std::vector<uint64_t> time_multiplex;

            /* Stats for potentials */
            std::vector<std::vector<double>> work_reduction;
            std::vector<std::vector<double>> speedup;
            std::vector<std::vector<uint64_t>> bit_multiplications;
            std::vector<uint64_t> parallel_multiplications;

            /* Stats for sparsity */
            std::vector<double> act_sparsity;
            std::vector<uint64_t> zero_act;
            std::vector<uint64_t> total_act;
            std::vector<double> wgt_sparsity;
            std::vector<uint64_t> zero_wgt;
            std::vector<uint64_t> total_wgt;

			/* Stats for training sparsity */
            std::vector<std::vector<double>> fw_act_sparsity;
            std::vector<std::vector<uint64_t>> fw_zero_act;
            std::vector<std::vector<uint64_t>> fw_total_act;
            std::vector<std::vector<double>> fw_wgt_sparsity;
            std::vector<std::vector<uint64_t>> fw_zero_wgt;
            std::vector<std::vector<uint64_t>> fw_total_wgt;

            std::vector<std::vector<double>> bw_in_grad_sparsity;
            std::vector<std::vector<uint64_t>> bw_zero_in_grad;
            std::vector<std::vector<uint64_t>> bw_total_in_grad;
            std::vector<std::vector<double>> bw_wgt_grad_sparsity;
            std::vector<std::vector<uint64_t>> bw_zero_wgt_grad;
            std::vector<std::vector<uint64_t>> bw_total_wgt_grad;
            std::vector<std::vector<double>> bw_out_grad_sparsity;
            std::vector<std::vector<uint64_t>> bw_zero_out_grad;
            std::vector<std::vector<uint64_t>> bw_total_out_grad;

            /* Training value distribution */
            bool mantissa_data;
            std::vector<std::vector<std::vector<uint64_t>>> fw_act_values;
            std::vector<std::vector<std::vector<uint64_t>>> fw_wgt_values;
            std::vector<std::vector<std::vector<uint64_t>>> bw_in_grad_values;
            std::vector<std::vector<std::vector<uint64_t>>> bw_wgt_grad_values;
            std::vector<std::vector<std::vector<uint64_t>>> bw_out_grad_values;

            /* Stats for average width */
            std::vector<std::vector<double>> act_avg_width;
            std::vector<std::vector<double>> act_width_reduction;
            std::vector<std::vector<std::vector<double>>> act_width_need;
            std::vector<std::vector<uint64_t>> act_bits_baseline;
            std::vector<std::vector<uint64_t>> act_bits_profiled;
            std::vector<std::vector<uint64_t>> act_bits_datawidth;
            std::vector<std::vector<uint64_t>> act_bits_scnn;
            std::vector<std::vector<double>> wgt_avg_width;
            std::vector<std::vector<double>> wgt_width_reduction;
            std::vector<std::vector<std::vector<double>>> wgt_width_need;
            std::vector<std::vector<uint64_t>> wgt_bits_baseline;
            std::vector<std::vector<uint64_t>> wgt_bits_profiled;
            std::vector<std::vector<uint64_t>> wgt_bits_datawidth;
            std::vector<std::vector<uint64_t>> wgt_bits_scnn;

            /* Stats for training average width */
            std::vector<std::vector<double>> fw_act_avg_width;
            std::vector<std::vector<uint64_t>> fw_act_bits_baseline;
            std::vector<std::vector<uint64_t>> fw_act_bits_datawidth;
            std::vector<std::vector<double>> fw_wgt_avg_width;
            std::vector<std::vector<uint64_t>> fw_wgt_bits_baseline;
            std::vector<std::vector<uint64_t>> fw_wgt_bits_datawidth;

            std::vector<std::vector<double>> bw_in_grad_avg_width;
            std::vector<std::vector<uint64_t>> bw_in_grad_bits_baseline;
            std::vector<std::vector<uint64_t>> bw_in_grad_bits_datawidth;
            std::vector<std::vector<double>> bw_wgt_grad_avg_width;
            std::vector<std::vector<uint64_t>> bw_wgt_grad_bits_baseline;
            std::vector<std::vector<uint64_t>> bw_wgt_grad_bits_datawidth;
            std::vector<std::vector<double>> bw_out_grad_avg_width;
            std::vector<std::vector<uint64_t>> bw_out_grad_bits_baseline;
            std::vector<std::vector<uint64_t>> bw_out_grad_bits_datawidth;


            template <typename T>
            T get_average(const std::vector<T> &vector_stat) const {
                return accumulate(vector_stat.begin(), vector_stat.end(), 0.0) / vector_stat.size();
            }

            template <typename T>
            T get_average(const std::vector<std::vector<T>> &vector_stat, bool skip_first = false) const {
                std::vector<T> averages = std::vector<T>(vector_stat.size() - skip_first,0);
                for(int i = skip_first; i < vector_stat.size(); i++) {
                    averages[i - skip_first] = this->get_average(vector_stat[i]);
                }
                return this->get_average(averages);
            }

            template <typename T>
            T get_total(const std::vector<T> &vector_stat) const {
                return accumulate(vector_stat.begin(), vector_stat.end(), 0.0);
            }

            template <typename T>
            T get_total(const std::vector<std::vector<T>> &vector_stat) const {
                std::vector<T> averages = std::vector<T>(vector_stat.size(),0);
                for(int i = 0; i < vector_stat.size(); i++) {
                    averages[i] = this->get_average(vector_stat[i]);
                }
                return this->get_total(averages);
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

        /* Update flags for the last stats
         * @param TENSORFLOW_8b     Tensorflow 8b network
         */
        static void updateFlagsLastStat(bool TENSORFLOW_8b);

        /* Getter */
        static const std::vector<Stats> &getAll_stats();

    };

}

#endif //DNNSIM_STATISTICS_H
