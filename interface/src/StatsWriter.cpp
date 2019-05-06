
#include <interface/StatsWriter.h>

namespace interface {

    void StatsWriter::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    void dump_csv_BitPragmatic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,stall_cycles,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,%lu,0\n", stats.layers[i].c_str(),j,stats.cycles[i][j],
                        stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/stats.cycles[i][j], 
                        stats.stall_cycles[i][j]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%lu,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i], (double)stats.baseline_cycles[i] /
                    stats.get_average(stats.cycles[i]), stats.get_average(stats.stall_cycles[i]),stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,%lu,%.2f\n", stats.get_total(stats.cycles),
                stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                (double)stats.get_total(stats.cycles),stats.get_total(stats.stall_cycles),total_time);
        o_file << line;
    }

    void dump_csv_Stripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,columns_per_act,rows_per_wgt,idle_columns,idle_rows,"
                  "act_precision,wgt_precision,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,%lu,%lu,%lu,%lu,%d,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/
                        stats.cycles[i][j], stats.columns_per_act[i], stats.rows_per_wgt[i], stats.idle_columns[i],
                        stats.idle_rows[i], stats.act_prec[i], stats.wgt_prec[i]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%lu,%lu,%lu,%lu,%d,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/
                    stats.get_average(stats.cycles[i]), stats.columns_per_act[i], stats.rows_per_wgt[i],
                    stats.idle_columns[i], stats.idle_rows[i], stats.act_prec[i], stats.wgt_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,%lu,%lu,%lu,%lu,-,-,%.2f\n", stats.get_total(stats.cycles),
                stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                (double)stats.get_total(stats.cycles), stats.get_average(stats.columns_per_act),
                stats.get_average(stats.rows_per_wgt), stats.get_average(stats.idle_columns),
                stats.get_average(stats.idle_rows),total_time);
        o_file << line;
    }

    void dump_csv_DynamicStripes_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,baseline_cycles,speedup,stall_cycles,rows_per_wgt,idle_rows,act_precision,"
                  "wgt_precision,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%.2f,%lu,%lu,%lu,%d,%d,0\n", stats.layers[i].c_str(), j,
                        stats.cycles[i][j], stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/
                        stats.cycles[i][j], stats.stall_cycles[i][j], stats.rows_per_wgt[i], stats.idle_rows[i],
                        stats.act_prec[i], stats.wgt_prec[i]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f,%lu,%lu,%lu,%d,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.baseline_cycles[i], (double)stats.baseline_cycles[i]/
                    stats.get_average(stats.cycles[i]), stats.get_average(stats.stall_cycles[i]), stats.rows_per_wgt[i],
                    stats.idle_rows[i], stats.act_prec[i], stats.wgt_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f,%lu,%lu,%lu,-,-,%.2f\n", stats.get_total(stats.cycles),
                stats.get_total(stats.baseline_cycles),stats.get_total(stats.baseline_cycles)/
                (double)stats.get_total(stats.cycles), stats.get_total(stats.stall_cycles),
                stats.get_average(stats.rows_per_wgt), stats.get_average(stats.idle_rows), total_time);
        o_file << line;
    }

    void dump_csv_Loom_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }
        #endif

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

    void dump_csv_Laconic_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j]);
                o_file << line;
            }
        }
        #endif

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
        o_file << "layer,n_act,cycles,stall_cycles,time(s)" << std::endl;


        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.stall_cycles[i][j]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.get_average(stats.stall_cycles[i]),stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%.2f\n", stats.get_total(stats.cycles),
                stats.get_total(stats.stall_cycles), total_time);
        o_file << line;
    }

    void dump_csv_BitTacticalP_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,stall_cycles,act_precision,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%d,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.stall_cycles[i][j], stats.act_prec[i]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.get_average(stats.stall_cycles[i]), stats.act_prec[i],
                    stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,-,%.2f\n", stats.get_total(stats.cycles),
                stats.get_total(stats.stall_cycles), total_time);
        o_file << line;
    }

    void dump_csv_SCNN_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_pe,idle_halo,"
                  "total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,accumulator_updates,i_loop,f_loop,"
                  "offchip_weight_reads,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
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
        #endif

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
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_column_cycles,column_stalls,"
                  "idle_pe,idle_halo,total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,"
                  "accumulator_updates,i_loop,f_loop,offchip_weight_reads,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[512];
                snprintf(line, sizeof(line),"%s,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
                        stats.layers[i].c_str(), j, stats.cycles[i][j], stats.dense_cycles[i][j], stats.mults[i][j],
                        stats.idle_bricks[i][j], stats.idle_conflicts[i][j], stats.idle_column_cycles[i][j],
                        stats.column_stalls[i][j], stats.idle_pe[i][j], stats.idle_halo[i][j],
                        stats.total_mult_cycles[i][j], stats.halo_transfers[i][j], stats.weight_buff_reads[i][j],
                        stats.act_buff_reads[i][j], stats.accumulator_updates[i][j], stats.i_loop[i][j],
                        stats.f_loop[i][j], stats.offchip_weight_reads[i][j]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[512];
            snprintf(line, sizeof(line),"%s,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                    stats.layers[i].c_str(), stats.get_average(stats.cycles[i]),
                    stats.get_average(stats.dense_cycles[i]), stats.get_average(stats.mults[i]),
                    stats.get_average(stats.idle_bricks[i]), stats.get_average(stats.idle_conflicts[i]),
                    stats.get_average(stats.idle_column_cycles[i]), stats.get_average(stats.column_stalls[i]),
                    stats.get_average(stats.idle_pe[i]), stats.get_average(stats.idle_halo[i]),
                    stats.get_average(stats.total_mult_cycles[i]), stats.get_average(stats.halo_transfers[i]),
                    stats.get_average(stats.weight_buff_reads[i]), stats.get_average(stats.act_buff_reads[i]),
                    stats.get_average(stats.accumulator_updates[i]), stats.get_average(stats.i_loop[i]),
                    stats.get_average(stats.f_loop[i]), stats.get_average(stats.offchip_weight_reads[i]),
                    stats.time[i].count());
            o_file << line;
        }

        char line[512];
        snprintf(line, sizeof(line),"TOTAL,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                stats.get_total(stats.cycles), stats.get_total(stats.dense_cycles), stats.get_total(stats.mults),
                stats.get_total(stats.idle_bricks), stats.get_total(stats.idle_conflicts),
                stats.get_total(stats.idle_column_cycles), stats.get_total(stats.column_stalls),
                stats.get_total(stats.idle_pe), stats.get_total(stats.idle_halo),
                stats.get_total(stats.total_mult_cycles), stats.get_total(stats.halo_transfers),
                stats.get_total(stats.weight_buff_reads), stats.get_total(stats.act_buff_reads),
                stats.get_total(stats.accumulator_updates),stats.get_total(stats.i_loop), stats.get_total(stats.f_loop),
                stats.get_total(stats.offchip_weight_reads),total_time);
        o_file << line;
    }

    void dump_csv_SCNNe_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,dense_cycles,mults,idle_bricks,idle_conflicts,idle_column_cycles,column_stalls,"
                  "idle_pe,idle_halo,total_mult_cycles,halo_transfers,weight_buff_reads,act_buff_reads,"
                  "accumulator_updates,i_loop,f_loop,offchip_weight_reads,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[512];
                snprintf(line, sizeof(line),"%s,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,0\n",
                        stats.layers[i].c_str(), j, stats.cycles[i][j], stats.dense_cycles[i][j], stats.mults[i][j],
                        stats.idle_bricks[i][j], stats.idle_conflicts[i][j], stats.idle_column_cycles[i][j],
                        stats.column_stalls[i][j], stats.idle_pe[i][j], stats.idle_halo[i][j],
                        stats.total_mult_cycles[i][j], stats.halo_transfers[i][j], stats.weight_buff_reads[i][j],
                        stats.act_buff_reads[i][j], stats.accumulator_updates[i][j], stats.i_loop[i][j],
                        stats.f_loop[i][j], stats.offchip_weight_reads[i][j]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[512];
            snprintf(line, sizeof(line),"%s,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                    stats.layers[i].c_str(), stats.get_average(stats.cycles[i]),
                    stats.get_average(stats.dense_cycles[i]), stats.get_average(stats.mults[i]),
                    stats.get_average(stats.idle_bricks[i]), stats.get_average(stats.idle_conflicts[i]),
                    stats.get_average(stats.idle_column_cycles[i]), stats.get_average(stats.column_stalls[i]),
                    stats.get_average(stats.idle_pe[i]), stats.get_average(stats.idle_halo[i]),
                    stats.get_average(stats.total_mult_cycles[i]), stats.get_average(stats.halo_transfers[i]),
                    stats.get_average(stats.weight_buff_reads[i]), stats.get_average(stats.act_buff_reads[i]),
                    stats.get_average(stats.accumulator_updates[i]), stats.get_average(stats.i_loop[i]),
                    stats.get_average(stats.f_loop[i]), stats.get_average(stats.offchip_weight_reads[i]),
                    stats.time[i].count());
            o_file << line;
        }

        char line[512];
        snprintf(line, sizeof(line),"TOTAL,AVG,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f\n",
                stats.get_total(stats.cycles), stats.get_total(stats.dense_cycles), stats.get_total(stats.mults),
                stats.get_total(stats.idle_bricks), stats.get_total(stats.idle_conflicts),
                stats.get_total(stats.idle_column_cycles), stats.get_total(stats.column_stalls),
                stats.get_total(stats.idle_pe), stats.get_total(stats.idle_halo),
                stats.get_total(stats.total_mult_cycles), stats.get_total(stats.halo_transfers),
                stats.get_total(stats.weight_buff_reads), stats.get_total(stats.act_buff_reads),
                stats.get_total(stats.accumulator_updates),stats.get_total(stats.i_loop), stats.get_total(stats.f_loop),
                stats.get_total(stats.offchip_weight_reads),total_time);
        o_file << line;
    }

    void dump_csv_BitFusion_cycles(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,cycles,perf_factor,time_multiplex,act_precision,wgt_precision,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.cycles.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%lu,%lu,%lu,%d,%d,0\n", stats.layers[i].c_str(), j, stats.cycles[i][j],
                        stats.perf_factor[i], stats.time_multiplex[i], stats.act_prec[i], stats.wgt_prec[i]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%lu,%lu,%lu,%d,%d,%.2f\n", stats.layers[i].c_str(),
                    stats.get_average(stats.cycles[i]), stats.perf_factor[i], stats.time_multiplex[i],
                    stats.act_prec[i], stats.wgt_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%lu,%lu,%lu,-,-,%.2f\n", stats.get_total(stats.cycles),
                stats.get_average(stats.perf_factor), stats.get_average(stats.time_multiplex), total_time);
        o_file << line;
    }

    void dump_csv_potentials(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,work_reduction,speedup,parallel_mult,bit_mult,act_precision,wgt_precision,time(s)"
               << std::endl;

        #ifdef PER_IMAGE_RESULTS
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
        #endif

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

    void dump_csv_sparsity(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,act_sparsity,zeros,total,wgt_sparsity,zeros,total" << std::endl;
        for (int i = 0; i < stats.layers.size(); i++) {
            char line[256];
            snprintf(line, sizeof(line), "%s,%.2f,%lu,%lu,%.2f,%lu,%lu\n", stats.layers[i].c_str(),
                    stats.act_sparsity[i], stats.zero_act[i], stats.total_act[i], stats.wgt_sparsity[i],
                    stats.zero_wgt[i], stats.total_wgt[i]);
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,%.2f,%lu,%lu,%.2f,%lu,%lu\n", stats.get_average(stats.act_sparsity),
                stats.get_total(stats.zero_act), stats.get_total(stats.total_act),
                stats.get_average(stats.wgt_sparsity), stats.get_total(stats.zero_wgt),
                stats.get_total(stats.total_wgt));
        o_file << line;
    }

    void dump_csv_average_width(std::ofstream &o_file, const sys::Statistics::Stats &stats) {
        o_file << "layer,n_act,act_avg_width,act_reduction,act_bits_baseline,act_bits_profiled,act_bits_datawidth,"
                  "act_bits_scnn,act_precision,wgt_avg_width,wgt_reduction,wgt_bits_baseline,wgt_bits_profiled,"
                  "wgt_bits_datawidth,wgt_bits_scnn,wgt_precision,time(s)" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.act_avg_width.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%d,%.2f,%.2f,%lu,%lu,%lu,%lu,%d,%.2f,%.2f,%lu,%lu,%lu,%lu,%d,0\n",
                        stats.layers[i].c_str(), j, stats.act_avg_width[i][j], stats.act_width_reduction[i][j],
                        stats.act_bits_baseline[i][j], stats.act_bits_profiled[i][j], stats.act_bits_datawidth[i][j],
                        stats.act_bits_scnn[i][j], stats.act_prec[i], stats.wgt_avg_width[i][j],
                        stats.wgt_width_reduction[i][j], stats.wgt_bits_baseline[i][j], stats.wgt_bits_profiled[i][j],
                        stats.wgt_bits_scnn[i][j], stats.wgt_bits_datawidth[i][j], stats.wgt_prec[i]);
                o_file << line;
            }
        }
        #endif

        double total_time = 0.;
        for (int i = 0; i < stats.layers.size(); i++) {
            total_time += stats.time[i].count();
            char line[256];
            snprintf(line, sizeof(line), "%s,AVG,%.2f,%.2f,%lu,%lu,%lu,%lu,%d,%.2f,%.2f,%lu,%lu,%lu,%lu,%d,%.2f\n",
                    stats.layers[i].c_str(), stats.get_average(stats.act_avg_width[i]),
                    stats.get_average(stats.act_width_reduction[i]), stats.get_average(stats.act_bits_baseline[i]),
                    stats.get_average(stats.act_bits_profiled[i]), stats.get_average(stats.act_bits_datawidth[i]),
                    stats.get_average(stats.act_bits_scnn[i]), stats.act_prec[i],
                    stats.get_average(stats.wgt_avg_width[i]), stats.get_average(stats.wgt_width_reduction[i]),
                    stats.get_average(stats.wgt_bits_baseline[i]), stats.get_average(stats.wgt_bits_profiled[i]),
                    stats.get_average(stats.wgt_bits_datawidth[i]), stats.get_average(stats.wgt_bits_scnn[i]),
                    stats.wgt_prec[i], stats.time[i].count());
            o_file << line;
        }

        char line[256];
        snprintf(line, sizeof(line), "TOTAL,AVG,%.2f,%.2f,%lu,%lu,%lu,%lu,-,%.2f,%.2f,%lu,%lu,%lu,%lu,-,%.2f\n",
                stats.get_average(stats.act_avg_width), stats.get_average(stats.act_width_reduction),
                stats.get_total(stats.act_bits_baseline), stats.get_total(stats.act_bits_profiled),
                stats.get_total(stats.act_bits_datawidth), stats.get_total(stats.act_bits_scnn),
                stats.get_average(stats.wgt_avg_width), stats.get_average(stats.wgt_width_reduction),
                stats.get_total(stats.wgt_bits_baseline), stats.get_total(stats.wgt_bits_profiled),
                stats.get_total(stats.wgt_bits_datawidth), stats.get_total(stats.wgt_bits_scnn), total_time);
        o_file << line;

        o_file << std::endl;
        o_file << "Activations" << std::endl;
        o_file << "layer,n_act,0 bits,1 bits,2 bits,3 bits,4 bits,5 bits,6 bits,7 bits,8 bits,9 bits,10 bits,11 bits,"
                  "12 bits,13 bits,14 bits,15 bits,16 bits" << std::endl;


        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.act_avg_width.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line2[512];
                snprintf(line2, sizeof(line2), "%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                        "%.2f,%.2f,%.2f,%.2f\n", stats.layers[i].c_str(), j, stats.act_width_need[0][i][j],
                        stats.act_width_need[1][i][j], stats.act_width_need[2][i][j], stats.act_width_need[3][i][j],
                        stats.act_width_need[4][i][j], stats.act_width_need[5][i][j], stats.act_width_need[6][i][j],
                        stats.act_width_need[7][i][j], stats.act_width_need[8][i][j], stats.act_width_need[9][i][j],
                        stats.act_width_need[10][i][j], stats.act_width_need[11][i][j], stats.act_width_need[12][i][j],
                        stats.act_width_need[13][i][j], stats.act_width_need[14][i][j], stats.act_width_need[15][i][j],
                        stats.act_width_need[16][i][j]);
                o_file << line2;
            }
        }
        #endif

        for (int i = 0; i < stats.layers.size(); i++) {
            char line2[512];
            snprintf(line2, sizeof(line2), "%s,AVG,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "%.2f,%.2f,%.2f,%.2f\n", stats.layers[i].c_str(), stats.get_average(stats.act_width_need[0][i]),
                    stats.get_average(stats.act_width_need[1][i]), stats.get_average(stats.act_width_need[2][i]),
                    stats.get_average(stats.act_width_need[3][i]), stats.get_average(stats.act_width_need[4][i]),
                    stats.get_average(stats.act_width_need[5][i]), stats.get_average(stats.act_width_need[6][i]),
                    stats.get_average(stats.act_width_need[7][i]), stats.get_average(stats.act_width_need[8][i]),
                    stats.get_average(stats.act_width_need[9][i]), stats.get_average(stats.act_width_need[10][i]),
                    stats.get_average(stats.act_width_need[11][i]), stats.get_average(stats.act_width_need[12][i]),
                    stats.get_average(stats.act_width_need[13][i]), stats.get_average(stats.act_width_need[14][i]),
                    stats.get_average(stats.act_width_need[15][i]), stats.get_average(stats.act_width_need[16][i]));
            o_file << line2;
        }

        char line2[512];
        snprintf(line2, sizeof(line2), "TOTAL,AVG,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                "%.2f,%.2f,%.2f,%.2f\n", stats.get_average(stats.act_width_need[0]),
                stats.get_average(stats.act_width_need[1]), stats.get_average(stats.act_width_need[2]),
                stats.get_average(stats.act_width_need[3]), stats.get_average(stats.act_width_need[4]),
                stats.get_average(stats.act_width_need[5]), stats.get_average(stats.act_width_need[6]),
                stats.get_average(stats.act_width_need[7]), stats.get_average(stats.act_width_need[8]),
                stats.get_average(stats.act_width_need[9]), stats.get_average(stats.act_width_need[10]),
                stats.get_average(stats.act_width_need[11]), stats.get_average(stats.act_width_need[12]),
                stats.get_average(stats.act_width_need[13]), stats.get_average(stats.act_width_need[14]),
                stats.get_average(stats.act_width_need[15]), stats.get_average(stats.act_width_need[16]));
        o_file << line2;

        o_file << std::endl;
        o_file << "Weights" << std::endl;
        o_file << "layer,n_act,0 bits,1 bits,2 bits,3 bits,4 bits,5 bits,6 bits,7 bits,8 bits,9 bits,10 bits,11 bits,"
                  "12 bits,13 bits,14 bits,15 bits,16 bits" << std::endl;

        #ifdef PER_IMAGE_RESULTS
        for (int j = 0; j < stats.act_avg_width.front().size(); j++) {
            for (int i = 0; i < stats.layers.size(); i++) {
                char line3[512];
                snprintf(line3, sizeof(line3), "%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                        "%.2f,%.2f,%.2f,%.2f\n", stats.layers[i].c_str(), j, stats.wgt_width_need[0][i][j],
                        stats.wgt_width_need[1][i][j], stats.wgt_width_need[2][i][j], stats.wgt_width_need[3][i][j],
                        stats.wgt_width_need[4][i][j], stats.wgt_width_need[5][i][j], stats.wgt_width_need[6][i][j],
                        stats.wgt_width_need[7][i][j], stats.wgt_width_need[8][i][j], stats.wgt_width_need[9][i][j],
                        stats.wgt_width_need[10][i][j], stats.wgt_width_need[11][i][j], stats.wgt_width_need[12][i][j],
                        stats.wgt_width_need[13][i][j], stats.wgt_width_need[14][i][j], stats.wgt_width_need[15][i][j],
                        stats.wgt_width_need[16][i][j]);
                o_file << line3;
            }
        }
        #endif

        for (int i = 0; i < stats.layers.size(); i++) {
            char line3[512];
            snprintf(line3, sizeof(line3), "%s,AVG,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "%.2f,%.2f,%.2f,%.2f\n", stats.layers[i].c_str(), stats.get_average(stats.wgt_width_need[0][i]),
                    stats.get_average(stats.wgt_width_need[1][i]), stats.get_average(stats.wgt_width_need[2][i]),
                    stats.get_average(stats.wgt_width_need[3][i]), stats.get_average(stats.wgt_width_need[4][i]),
                    stats.get_average(stats.wgt_width_need[5][i]), stats.get_average(stats.wgt_width_need[6][i]),
                    stats.get_average(stats.wgt_width_need[7][i]), stats.get_average(stats.wgt_width_need[8][i]),
                    stats.get_average(stats.wgt_width_need[9][i]), stats.get_average(stats.wgt_width_need[10][i]),
                    stats.get_average(stats.wgt_width_need[11][i]), stats.get_average(stats.wgt_width_need[12][i]),
                    stats.get_average(stats.wgt_width_need[13][i]), stats.get_average(stats.wgt_width_need[14][i]),
                    stats.get_average(stats.wgt_width_need[15][i]), stats.get_average(stats.wgt_width_need[16][i]));
            o_file << line3;
        }

        char line3[512];
        snprintf(line3, sizeof(line3), "TOTAL,AVG,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                "%.2f,%.2f,%.2f,%.2f\n", stats.get_average(stats.wgt_width_need[0]),
                stats.get_average(stats.wgt_width_need[1]), stats.get_average(stats.wgt_width_need[2]),
                stats.get_average(stats.wgt_width_need[3]), stats.get_average(stats.wgt_width_need[4]),
                stats.get_average(stats.wgt_width_need[5]), stats.get_average(stats.wgt_width_need[6]),
                stats.get_average(stats.wgt_width_need[7]), stats.get_average(stats.wgt_width_need[8]),
                stats.get_average(stats.wgt_width_need[9]), stats.get_average(stats.wgt_width_need[10]),
                stats.get_average(stats.wgt_width_need[11]), stats.get_average(stats.wgt_width_need[12]),
                stats.get_average(stats.wgt_width_need[13]), stats.get_average(stats.wgt_width_need[14]),
                stats.get_average(stats.wgt_width_need[15]), stats.get_average(stats.wgt_width_need[16]));
        o_file << line3;

    }

    void StatsWriter::dump_csv() {

        for(const sys::Statistics::Stats &stats : sys::Statistics::getAll_stats()) {
            std::ofstream o_file;
            check_path("results/" + stats.net_name);
            std::string path = "results/" + stats.net_name + "/" + stats.arch + "_" + stats.task_name;
            path += stats.tensorflow_8b ? "-TF.csv" : ".csv";
            o_file.open (path);
            o_file << stats.net_name << std::endl;
            o_file << stats.arch << std::endl;

            std::string arch = stats.arch.substr(0,stats.arch.find('_'));
            if(!stats.cycles.empty() && arch == "BitPragmatic") dump_csv_BitPragmatic_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "Stripes") dump_csv_Stripes_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "DynamicStripes") dump_csv_DynamicStripes_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "Loom") dump_csv_Loom_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "Laconic") dump_csv_Laconic_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitTacticalP") dump_csv_BitTacticalP_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitTacticalE") dump_csv_BitTacticalE_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNN") dump_csv_SCNN_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNNp") dump_csv_SCNNp_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "SCNNe") dump_csv_SCNNe_cycles(o_file,stats);
            else if(!stats.cycles.empty() && arch == "BitFusion") dump_csv_BitFusion_cycles(o_file,stats);
            else if(!stats.work_reduction.empty()) dump_csv_potentials(o_file,stats);
            else if(!stats.act_sparsity.empty()) dump_csv_sparsity(o_file,stats);
			else if(!stats.fw_act_sparsity.empty()) dump_csv_training_sparsity(o_file,stats);
            else if(!stats.act_avg_width.empty()) dump_csv_average_width(o_file,stats);

            o_file.close();
        }

    }

}
