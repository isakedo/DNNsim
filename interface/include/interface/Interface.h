#ifndef DNNSIM_INTERFACE_H
#define DNNSIM_INTERFACE_H

#include <sys/common.h>
#include <base/Network.h>
#include <network.pb.h>

typedef std::vector<std::vector<std::vector<std::tuple<int,int,int,uint16_t>>>> inf_schedule;
typedef std::vector<std::vector<std::tuple<int,int,int,uint16_t>>> inf_set_schedule;
typedef std::vector<std::tuple<int,int,int,uint16_t>> inf_time_schedule;
typedef std::tuple<int,int,int,uint16_t> inf_schedule_tuple;

namespace interface {

    /**
     * Interface base class
     */
    class Interface {

    protected:

        /** Avoid std::out messages */
        const bool QUIET;

        /** Check if the path exists
         * @param path  Path we want to check
         */
        void check_path(const std::string &path) {
            std::ifstream file(path);
            if(!file.good()) {
                throw std::runtime_error("The path " + path + " does not exist.");
            }
        }

        /** Constructor
         * @param _QUIET    Remove stdout messages
         */
        explicit Interface(bool _QUIET) : QUIET(_QUIET) {}

    };

}


#endif //DNNSIM_INTERFACE_H
