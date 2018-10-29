#ifndef DNNSIM_STATSWRITER_H
#define DNNSIM_STATSWRITER_H

#include <sys/common.h>
#include <sys/Statistics.h>

namespace interface {

        class StatsWriter {

        private:

            /* Statistics we want to dump */
            sys::Statistics stats;

            /* Path where we want to dump the statistics */
            std::string path;

        public:

            /* Constructor
             * @param _stats    Stats that we want to dump
             * @param _path     Path where we want to dump the statistics (If not suffix, it is added in the method)
             */
            StatsWriter(const sys::Statistics &_stats) : stats(_stats) {}

            /* Dump the statistics in a text file */
            void dump_txt();

            /* Dump the statistics in a csv file */
            void dump_csv();


        };

}


#endif //DNNSIM_STATSWRITER_H
