
#include <sys/Statistics.h>

namespace sys {

    std::vector<Statistics::Stats> Statistics::all_stats;

    void Statistics::initialize(Stats &stats) {
        stats.net_name = "";
        stats.arch = "";
    }

    void Statistics::addStats(const Stats &_stats) {
        Statistics::all_stats.push_back(_stats);
    }

    /* Getter */
    const std::vector<Statistics::Stats> &Statistics::getAll_stats() {
        return Statistics::all_stats;
    }

}