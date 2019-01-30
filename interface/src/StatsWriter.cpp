
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
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%u,%.2f,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%u,%.2f,%.2f\n", stats.layers[i].c_str(), stats.avg_cycles[i],
                     stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/stats.avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%u,%.2f,%.2f\n", stats.get_total(stats.avg_cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.avg_cycles),total_time);
        o_file << line;
    }

    void dump_csv_Stripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%u,%.2f,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i],
                        (double)stats.baseline_cycles[i]/stats.cycles[i][j], stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%u,%.2f,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.avg_cycles[i], stats.baseline_cycles[i],
                    (double)stats.baseline_cycles[i]/stats.avg_cycles[i], stats.act_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%u,%.2f,-,%.2f\n", stats.get_total(stats.avg_cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.avg_cycles),total_time);
        o_file << line;
    }

    void dump_csv_DynamicStripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%u,%.2f,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i],
                        (double)stats.baseline_cycles[i]/stats.cycles[i][j], stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%u,%.2f,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.avg_cycles[i], stats.baseline_cycles[i],
                    (double)stats.baseline_cycles[i]/stats.avg_cycles[i], stats.act_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%u,%.2f,-,%.2f\n", stats.get_total(stats.avg_cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.avg_cycles),total_time);
        o_file << line;
    }

    void dump_csv_Laconic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%.2f\n", stats.layers[i].c_str(), stats.avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%.2f\n", stats.get_total(stats.avg_cycles), total_time);
        o_file << line;
    }

    void dump_csv_BitTacticalE_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%.2f\n", stats.layers[i].c_str(), stats.avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%.2f\n", stats.get_total(stats.avg_cycles), total_time);
        o_file << line;
    }

    void dump_csv_BitTacticalP_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,%d,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%d,%.2f\n", stats.layers[i].c_str(), stats.avg_cycles[i],
                     stats.act_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,-,%.2f\n", stats.get_total(stats.avg_cycles), total_time);
        o_file << line;
    }

    void dump_csv_SCNN_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%u,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%u,%.2f\n", stats.layers[i].c_str(), stats.avg_cycles[i],
                     stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%u,%.2f\n", stats.get_total(stats.avg_cycles), total_time);
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

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%.2f,%.2f,%ld,%ld,-,-,%f\n",stats.get_average(stats.avg_work_reduction),
                stats.get_average(stats.avg_speedup),stats.get_total(stats.parallel_multiplications),
                stats.get_total(stats.avg_bit_multiplications), total_time);
        o_file << line;
    }

    void StatsWriter::dump_csv() {

        for(const sys::Statistics::Stats &stats : sys::Statistics::getAll_stats()) {
            std::ofstream o_file;
            check_path("results/" + stats.net_name);
            o_file.open ("results/" + stats.net_name + "/" + stats.arch + "_" + stats.task_name + ".csv");
            o_file << stats.net_name << std::endl;
            o_file << stats.arch << std::endl;

            std::string arch = stats.arch.substr(0,stats.arch.find('_'));
            if(!stats.cycles.empty() && arch == "BitPragmatic") dump_csv_BitPragmatic_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "Stripes") dump_csv_Stripes_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "DynamicStripes") dump_csv_DynamicStripes_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "Laconic") dump_csv_Laconic_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitTacticalE") dump_csv_BitTacticalE_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitTacticalP") dump_csv_BitTacticalP_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNN") dump_csv_SCNN_cycles(o_file,stats);
            else if(!stats.work_reduction.empty()) dump_csv_potentials(o_file,stats);

            o_file.close();
        }

    }

}