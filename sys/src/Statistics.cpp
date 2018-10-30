
#include <sys/Statistics.h>

namespace sys {

    void Statistics::addStats(const Stats &_stats) {
        all_stats.push_back(_stats);
    }

    /* Getter */
    const std::vector<Statistics::Stats> &Statistics::getAll_stats() {
        return all_stats;
    }

}