
#include <interface/StatsWriter.h>

namespace interface {

    void StatsWriter::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    void dump_csv_BitPragmatic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.PRA_cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%u,0\n", stats.layers[i].c_str(), j,
                         stats.PRA_cycles[i][j], stats.PRA_baseline_cycles[i]);
                o_file << line;
            }
        }
        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%u,%f\n", stats.layers[i].c_str(), stats.PRA_avg_cycles[i],
                     stats.PRA_baseline_cycles[i], stats.time[i].count());
            o_file << line;
        }
        auto total_cycles = accumulate(stats.PRA_avg_cycles.begin(), stats.PRA_avg_cycles.end(), 0.0);
        auto total_baseline_cycles = accumulate(stats.PRA_baseline_cycles.begin(),
                                                stats.PRA_baseline_cycles.end(), 0.0);
        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%u,%f\n", (uint32_t)total_cycles,
                 (uint32_t)total_baseline_cycles,total_time);
        o_file << line;
    }

    void dump_csv_potentials(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        //TODO fix statistics
        o_file << "layer,n_act,potentials,multiplications,one_bit_mult,act_precision,wgt_precision,time(s)"
               << std::endl;
        for (int j = 0; j < stats.potentials.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%.2f,%ld,%ld,%d,%d,0\n", stats.layers[i].c_str(), j,
                         stats.potentials[i][j], stats.multiplications[i],
                         stats.one_bit_multiplications[i][j],
                         stats.act_prec[i], stats.wgt_prec[i]);
                o_file << line;
            }
        }
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
            else if(!stats.potentials.empty()) dump_csv_potentials(o_file,stats);
            else if(!stats.on_chip_weights.empty()) dump_csv_mem_accesses(o_file,stats);

            o_file.close();
        }

    }

}