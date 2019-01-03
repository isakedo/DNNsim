
#include <interface/StatsWriter.h>

namespace interface {

    void StatsWriter::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    void dump_csv_BitPragmatic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,time(s)" << std::endl;
        for (int j = 0; j < stats.PRA_cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%u,%.2f,0\n", stats.layers[i].c_str(), j, stats.PRA_cycles[i][j],
                        stats.PRA_baseline_cycles[i], (double)stats.PRA_baseline_cycles[i]/stats.PRA_cycles[i][j]);
                o_file << line;
            }
        }
        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%u,%.2f,%.2f\n", stats.layers[i].c_str(), stats.PRA_avg_cycles[i],
                     stats.PRA_baseline_cycles[i], (double)stats.PRA_baseline_cycles[i]/stats.PRA_avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }
        auto total_cycles = accumulate(stats.PRA_avg_cycles.begin(), stats.PRA_avg_cycles.end(), 0.0);
        auto total_baseline_cycles = accumulate(stats.PRA_baseline_cycles.begin(),
                stats.PRA_baseline_cycles.end(), 0.0);
        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%u,%.2f,%.2f\n", (uint32_t)total_cycles,
                 (uint32_t)total_baseline_cycles,total_baseline_cycles/total_cycles,total_time);
        o_file << line;
    }

    void dump_csv_Laconic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.LAC_cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,0\n", stats.layers[i].c_str(), j, stats.LAC_cycles[i][j]);
                o_file << line;
            }
        }
        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%.2f\n", stats.layers[i].c_str(), stats.LAC_avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }
        auto total_cycles = accumulate(stats.LAC_avg_cycles.begin(), stats.LAC_avg_cycles.end(), 0.0);

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%.2f\n", (uint32_t)total_cycles, total_time);
        o_file << line;
    }

    void dump_csv_potentials(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,work_reduction,speedup,parallel_mult,bit_mult,act_precision,wgt_precision,time(s)"
               << std::endl;
        for (int j = 0; j < stats.work_reduction.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%.2f,%.2f,%ld,%ld,%d,%d,0\n", stats.layers[i].c_str(), j,
                         stats.work_reduction[i][j], stats.speedup[i][j], stats.parallel_multiplications[i],
                         stats.bit_multiplications[i][j],
                         stats.act_prec[i], stats.wgt_prec[i]);
                o_file << line;
            }
        }
        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%.2f,%.2f,%ld,%ld,%d,%d,%f\n", stats.layers[i].c_str(),
                    stats.avg_work_reduction[i], stats.avg_speedup[i], stats.parallel_multiplications[i],
                    stats.avg_bit_multiplications[i], stats.act_prec[i], stats.wgt_prec[i], stats.time[i].count());
            o_file << line;
        }
        auto avg_work_reduction = accumulate(stats.avg_work_reduction.begin(), stats.avg_work_reduction.end(), 0.0) /
                stats.avg_work_reduction.size();
        auto avg_speedup = accumulate(stats.avg_speedup.begin(), stats.avg_speedup.end(), 0.0) /
                stats.avg_speedup.size();
        auto total_bit_multiplications = (uint64_t)accumulate(stats.avg_bit_multiplications.begin(),
                stats.avg_bit_multiplications.end(), 0.0);
        auto total_parallel_multiplications = (uint64_t)accumulate(stats.parallel_multiplications.begin(),
                stats.parallel_multiplications.end(), 0.0);
        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%.2f,%.2f,%ld,%ld,-,-,%f\n", avg_work_reduction, avg_speedup,
                total_parallel_multiplications, total_bit_multiplications, total_time);
        o_file << line;
    }

    void dump_csv_mem_accesses(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,on_chip_weights,on_chip_activations,off_chip_weights_sch3,off_chip_weights_sch4,"
                  "off_chip_activations,bits_weights,bits_working_weights,bits_one_activations_row,computation"
               << std::endl;
        for (int i = 0; i < stats.layers.size(); i++) {
            char line[256];
            snprintf(line, sizeof(line), "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", stats.layers[i].c_str(),
                     stats.on_chip_weights[i], stats.on_chip_activations[i], stats.off_chip_weights_sch3[i],
                     stats.off_chip_weights_sch4[i], stats.off_chip_activations[i],stats.bits_weights[i],
                     stats.bits_working_weights[i],stats.bits_one_activation_row[i],stats.computations[i]);
            o_file << line;
        }
    }

    void StatsWriter::dump_csv() {

        for(const sys::Statistics::Stats &stats : sys::Statistics::getAll_stats()) {
            std::ofstream o_file;
            check_path("results/" + stats.net_name);
            o_file.open ("results/" + stats.net_name + "/" + stats.arch + "_" + stats.task_name + ".csv");
            o_file << stats.net_name << std::endl;
            o_file << stats.arch << std::endl;

            if(!stats.PRA_cycles.empty()) dump_csv_BitPragmatic_cycles(o_file,stats);
            else if(!stats.LAC_cycles.empty()) dump_csv_Laconic_cycles(o_file,stats);
            else if(!stats.work_reduction.empty()) dump_csv_potentials(o_file,stats);
            else if(!stats.on_chip_weights.empty()) dump_csv_mem_accesses(o_file,stats);

            o_file.close();
        }

    }

}