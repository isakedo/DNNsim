
#include <interface/StatsWriter.h>

namespace interface {

    void StatsWriter::dump_txt() {

        for(const sys::Statistics::Stats &stats : sys::Statistics::getAll_stats()) {

        }

    }

    void StatsWriter::dump_csv() {

        for(const sys::Statistics::Stats &stats : sys::Statistics::getAll_stats()) {
            std::ofstream o_file;
            o_file.open ("results/" + stats.net_name + "/stats.csv");
            o_file << stats.net_name << std::endl;
            o_file << stats.arch << std::endl;

            if(!stats.work_reduction.empty()) {
                o_file << "layer,n_act,work_reduction,multiplications,one_bit_mult,act_precision,wgt_precision"
                       << std::endl;
                for (int j = 0; j < stats.work_reduction.front().size(); j++) {
                    for (int i = 0; i < stats.layers.size(); i++) {
                        char line[256];
                        snprintf(line, sizeof(line), "%s,%d,%.2f,%ld,%ld,%d,%d\n", stats.layers[i].c_str(), j,
                                 stats.work_reduction[i][j], stats.multiplications[i],
                                 stats.one_bit_multiplications[i][j],
                                 stats.act_prec[i], stats.wgt_prec[i]);
                        o_file << line;
                    }
                }
            }

            if(!stats.on_chip_weights.empty()) {
                o_file << "layer,on_chip_weights,on_chip_activations,off_chip_weights_sch3,off_chip_weights_sch4,"
                          "off_chip_activations,bits_weights,bits_working_weights,bits_one_activations_row,computation"
                       << std::endl;
                for (int i = 0; i < stats.layers.size(); i++) {
                    char line[256];
                    snprintf(line, sizeof(line), "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", stats.layers[i].c_str(),
                            stats.on_chip_weights[i], stats.off_chip_activations[i], stats.off_chip_weights_sch3[i],
                            stats.off_chip_weights_sch4[i], stats.off_chip_activations[i],stats.bits_weights[i],
                            stats.bits_working_weights[i],stats.bits_one_activation_row[i],stats.computations[i]);
                    o_file << line;
                }
            }

            o_file.close();
        }

    }

}