
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
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i], (double)stats.baseline_cycles[i] /
                    stats.get_average(stats.cycles[i]), stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,%.2f\n", stats.get_total(stats.cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.cycles),total_time);
        o_file << line;
    }

    void dump_csv_Stripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i],
                        (double)stats.baseline_cycles[i]/stats.cycles[i][j], stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i],
                    (double)stats.baseline_cycles[i]/stats.get_average(stats.cycles[i]), stats.act_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,-,%.2f\n", stats.get_total(stats.cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.cycles),total_time);
        o_file << line;
    }

    void dump_csv_DynamicStripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i],
                        (double)stats.baseline_cycles[i]/stats.cycles[i][j], stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i],
                    (double)stats.baseline_cycles[i]/stats.get_average(stats.cycles[i]), stats.act_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,-,%.2f\n", stats.get_total(stats.cycles),
                 stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                         (double)stats.get_total(stats.cycles),total_time);
        o_file << line;
    }

    void dump_csv_Laconic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%.2f\n", stats.layers[i].c_str(),stats.get_average(stats.cycles[i])
                    , stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%.2f\n", stats.get_total(stats.cycles), total_time);
        o_file << line;
    }

    void dump_csv_BitTacticalE_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%.2f\n", stats.layers[i].c_str(),stats.get_average(stats.cycles[i])
                    , stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%.2f\n", stats.get_total(stats.cycles), total_time);
        o_file << line;
    }

    void dump_csv_BitTacticalP_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,act_precision,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%d,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.act_prec[i]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.act_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,-,%.2f\n", stats.get_total(stats.cycles), total_time);
        o_file << line;
    }

    void dump_csv_SCNN_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_pe,idle_halo,"
                  "total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,accumulator_updates,i_loop,f_loop,"
                  "offchip_weight_reads,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[512];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
                        stats.layers[i].c_str(), j, stats.cycles[i][j], stats.dense_cycles[i][j], stats.mults[i][j],
                        stats.idle_bricks[i][j], stats.idle_conflicts[i][j], stats.idle_pe[i][j], stats.idle_halo[i][j],
                        stats.total_mult_cycles[i][j], stats.halo_transfers[i][j], stats.weight_buff_reads[i][j],
                        stats.act_buff_reads[i][j], stats.accumulator_updates[i][j], stats.i_loop[i][j],
                        stats.f_loop[i][j], stats.offchip_weight_reads[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[512];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                    stats.layers[i].c_str(), stats.get_average(stats.cycles[i]),
                    stats.get_average(stats.dense_cycles[i]), stats.get_average(stats.mults[i]),
                    stats.get_average(stats.idle_bricks[i]), stats.get_average(stats.idle_conflicts[i]),
                    stats.get_average(stats.idle_pe[i]), stats.get_average(stats.idle_halo[i]),
                    stats.get_average(stats.total_mult_cycles[i]), stats.get_average(stats.halo_transfers[i]),
                    stats.get_average(stats.weight_buff_reads[i]), stats.get_average(stats.act_buff_reads[i]),
                    stats.get_average(stats.accumulator_updates[i]), stats.get_average(stats.i_loop[i]),
                    stats.get_average(stats.f_loop[i]), stats.get_average(stats.offchip_weight_reads[i]),
                    stats.time[i].count());
            o_file << line;
        }

        char line[512];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                stats.get_total(stats.cycles), stats.get_total(stats.dense_cycles), stats.get_total(stats.mults),
                stats.get_total(stats.idle_bricks), stats.get_total(stats.idle_conflicts),
                stats.get_total(stats.idle_pe), stats.get_total(stats.idle_halo),
                stats.get_total(stats.total_mult_cycles), stats.get_total(stats.halo_transfers),
                stats.get_total(stats.weight_buff_reads), stats.get_total(stats.act_buff_reads),
                stats.get_total(stats.accumulator_updates),stats.get_total(stats.i_loop), stats.get_total(stats.f_loop),
                stats.get_total(stats.offchip_weight_reads),total_time);
        o_file << line;
    }

    void dump_csv_SCNNp_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_pe,idle_halo,"
                  "total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,accumulator_updates,i_loop,f_loop,"
                  "offchip_weight_reads,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[512];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
                         stats.layers[i].c_str(), j, stats.cycles[i][j], stats.dense_cycles[i][j], stats.mults[i][j],
                         stats.idle_bricks[i][j], stats.idle_conflicts[i][j], stats.idle_pe[i][j], stats.idle_halo[i][j],
                         stats.total_mult_cycles[i][j], stats.halo_transfers[i][j], stats.weight_buff_reads[i][j],
                         stats.act_buff_reads[i][j], stats.accumulator_updates[i][j], stats.i_loop[i][j],
                         stats.f_loop[i][j], stats.offchip_weight_reads[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[512];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                     stats.layers[i].c_str(), stats.get_average(stats.cycles[i]),
                     stats.get_average(stats.dense_cycles[i]), stats.get_average(stats.mults[i]),
                     stats.get_average(stats.idle_bricks[i]), stats.get_average(stats.idle_conflicts[i]),
                     stats.get_average(stats.idle_pe[i]), stats.get_average(stats.idle_halo[i]),
                     stats.get_average(stats.total_mult_cycles[i]), stats.get_average(stats.halo_transfers[i]),
                     stats.get_average(stats.weight_buff_reads[i]), stats.get_average(stats.act_buff_reads[i]),
                     stats.get_average(stats.accumulator_updates[i]), stats.get_average(stats.i_loop[i]),
                     stats.get_average(stats.f_loop[i]), stats.get_average(stats.offchip_weight_reads[i]),
                     stats.time[i].count());
            o_file << line;
        }

        char line[512];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                 stats.get_total(stats.cycles), stats.get_total(stats.dense_cycles), stats.get_total(stats.mults),
                 stats.get_total(stats.idle_bricks), stats.get_total(stats.idle_conflicts),
                 stats.get_total(stats.idle_pe), stats.get_total(stats.idle_halo),
                 stats.get_total(stats.total_mult_cycles), stats.get_total(stats.halo_transfers),
                 stats.get_total(stats.weight_buff_reads), stats.get_total(stats.act_buff_reads),
                 stats.get_total(stats.accumulator_updates),stats.get_total(stats.i_loop), stats.get_total(stats.f_loop),
                 stats.get_total(stats.offchip_weight_reads),total_time);
        o_file << line;
    }

    void dump_csv_SCNNe_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_pe,idle_halo,"
                  "total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,accumulator_updates,i_loop,f_loop,"
                  "offchip_weight_reads,time(s)" << std::endl;
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[512];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
                         stats.layers[i].c_str(), j, stats.cycles[i][j], stats.dense_cycles[i][j], stats.mults[i][j],
                         stats.idle_bricks[i][j], stats.idle_conflicts[i][j], stats.idle_pe[i][j], stats.idle_halo[i][j],
                         stats.total_mult_cycles[i][j], stats.halo_transfers[i][j], stats.weight_buff_reads[i][j],
                         stats.act_buff_reads[i][j], stats.accumulator_updates[i][j], stats.i_loop[i][j],
                         stats.f_loop[i][j], stats.offchip_weight_reads[i][j]);
                o_file << line;
            }
        }

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[512];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                     stats.layers[i].c_str(), stats.get_average(stats.cycles[i]),
                     stats.get_average(stats.dense_cycles[i]), stats.get_average(stats.mults[i]),
                     stats.get_average(stats.idle_bricks[i]), stats.get_average(stats.idle_conflicts[i]),
                     stats.get_average(stats.idle_pe[i]), stats.get_average(stats.idle_halo[i]),
                     stats.get_average(stats.total_mult_cycles[i]), stats.get_average(stats.halo_transfers[i]),
                     stats.get_average(stats.weight_buff_reads[i]), stats.get_average(stats.act_buff_reads[i]),
                     stats.get_average(stats.accumulator_updates[i]), stats.get_average(stats.i_loop[i]),
                     stats.get_average(stats.f_loop[i]), stats.get_average(stats.offchip_weight_reads[i]),
                     stats.time[i].count());
            o_file << line;
        }

        char line[512];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                 stats.get_total(stats.cycles), stats.get_total(stats.dense_cycles), stats.get_total(stats.mults),
                 stats.get_total(stats.idle_bricks), stats.get_total(stats.idle_conflicts),
                 stats.get_total(stats.idle_pe), stats.get_total(stats.idle_halo),
                 stats.get_total(stats.total_mult_cycles), stats.get_total(stats.halo_transfers),
                 stats.get_total(stats.weight_buff_reads), stats.get_total(stats.act_buff_reads),
                 stats.get_total(stats.accumulator_updates),stats.get_total(stats.i_loop), stats.get_total(stats.f_loop),
                 stats.get_total(stats.offchip_weight_reads),total_time);
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
                     stats.get_average(stats.work_reduction[i]), stats.get_average(stats.speedup[i]),
                     stats.parallel_multiplications[i], stats.get_average(stats.bit_multiplications[i]),
                     stats.act_prec[i], stats.wgt_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%.2f,%.2f,%ld,%ld,-,-,%f\n",stats.get_average(stats.work_reduction),
                stats.get_average(stats.speedup),stats.get_total(stats.parallel_multiplications),
                stats.get_total(stats.bit_multiplications), total_time);
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
            else if(!stats.cycles.empty() && arch == "BitTacticalP") dump_csv_BitTacticalP_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitTacticalE") dump_csv_BitTacticalE_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNN") dump_csv_SCNN_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNNp") dump_csv_SCNNp_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNNe") dump_csv_SCNNe_cycles(o_file,stats);
            else if(!stats.work_reduction.empty()) dump_csv_potentials(o_file,stats);

            o_file.close();
        }

    }

}