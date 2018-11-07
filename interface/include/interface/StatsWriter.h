#ifndef DNNSIM_STATSWRITER_H
#define DNNSIM_STATSWRITER_H

#include <sys/common.h>
#include <sys/Statistics.h>

namespace interface {

        class StatsWriter {

        public:

            /* Dump the statistics in a text file */
            static void dump_txt();

            /* Dump the statistics in a csv file */
            static void dump_csv();

        };

}


#endif //DNNSIM_STATSWRITER_H
