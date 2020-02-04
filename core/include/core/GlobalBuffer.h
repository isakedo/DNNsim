#ifndef DNNSIM_GLOBALBUFFER_H
#define DNNSIM_GLOBALBUFFER_H

#include <sys/common.h>

namespace core {

    /**
     *
     */
    class GlobalBuffer {

    private:

        struct MemoryEntry {
            int bank = -1;
            bool on_chip = false;
        };

        uint64_t read_delay = 0;

        uint64_t write_delay = 0;

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
