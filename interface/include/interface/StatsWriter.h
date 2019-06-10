#ifndef DNNSIM_STATSWRITER_H
#define DNNSIM_STATSWRITER_H

#include "Interface.h"
#include <sys/Statistics.h>
#include <sys/stat.h>

#define PER_IMAGE_RESULTS
//#define PER_EPOCH_RESULTS

namespace interface {

        class StatsWriter : public Interface {

        public:

            /* Constructor
             * @param _QUIET    Remove stdout messages
             */
            explicit StatsWriter(bool _QUIET) : Interface(_QUIET) {}

            /* Dump the statistics in a csv file */
            void dump_csv();

        };

}


#endif //DNNSIM_STATSWRITER_H
