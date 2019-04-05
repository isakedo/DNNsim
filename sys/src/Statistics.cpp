
#include <sys/Statistics.h>

namespace sys {

    std::vector<Statistics::Stats> Statistics::all_stats;

    void Statistics::initialize(Stats &stats) {
        stats.task_name = "none";
        stats.net_name = "none";
        stats.arch = "none";
    }

    void Statistics::addStats(const Stats &_stats) {
        Statistics::all_stats.push_back(_stats);
    }

    void Statistics::updateFlagsLastStat(bool TENSORFLOW_8b) {
        Statistics::all_stats.back().tensorflow_8b = TENSORFLOW_8b;
    }


    /* Getter */
    const std::vector<Statistics::Stats> &Statistics::getAll_stats() {
        return Statistics::all_stats;
    }

}