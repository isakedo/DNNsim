#ifndef DNNSIM_INTERFACE_H
#define DNNSIM_INTERFACE_H

#include <sys/common.h>
#include <base/Network.h>
#include <network.pb.h>
#include <schedule.pb.h>

namespace interface {

    class Interface {

    protected:

        const bool QUIET;

        /* Check if the path exists
         * @param path  Path we want to check
         */
        void check_path(const std::string &path) {
            std::ifstream file(path);
            if(!file.good()) {
                throw std::runtime_error("The path " + path + " does not exist.");
            }
        }

        /* Constructor
         * @param _QUIET    Remove stdout messages
         */
        explicit Interface(bool _QUIET) : QUIET(_QUIET) {}

    };

}


#endif //DNNSIM_INTERFACE_H
