
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
            o_file << "layer,work_reduction,multiplications,effectual_bits,act_precision,wgt_precision" << std::endl;
            for(int i = 0; i < stats.layers.size(); i++) {
                char line[256];
                snprintf(line, sizeof(line), "%s,%.2f,%ld,%ld,%d,%d\n", stats.layers[i].c_str(),stats.work_reduction[i],
                        stats.multiplications[i],stats.effectual_bits[i],stats.act_prec[i],stats.wgt_prec[i]);
                o_file << line;
            }
            o_file.close();
        }

    }

}