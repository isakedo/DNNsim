#ifndef DNNSIM_NETWRITER_H
#define DNNSIM_NETWRITER_H

#include "Interface.h"

namespace interface {

    template <typename T>
    class NetWriter : public Interface {

    private:

        /* Name of the network */
        std::string name;

        /* Store a layer of the network into a protobuf layer
         * @param layer_proto   Pointer to a protobuf layer
         * @param layer         Layer of the network that want to be stored
         */
        void fill_layer(protobuf::Network_Layer* layer_proto, const base::Layer<T> &layer);

        /* Store the a tuple of the scheduler into a protobuf tuple
         * @param schedule_tuple_proto  Schedule tuple for protobuf
         * @param dense_schedule_tuple  Schedule tuple
         */
        void fill_schedule_tuple(protobuf::Schedule_Layer_Set_Time_Tuple* schedule_tuple_proto,
                const schedule_tuple &dense_schedule_tuple);

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _QUIET    Remove stdout messages
         */
        NetWriter(const std::string &_name, bool _QUIET) : Interface(_QUIET) {
            this->name = _name;
        }

        /* Store the network in protobuf format
         * @param network       Network that want to be stored
         */
        void write_network_protobuf(const base::Network<T> &network);

        /* Store the scheduler in protobuf format
         * @param network_schedule  Network schedule that want to be stored
         * @param schedule_type     Identify the type of schedule
         */
        void write_schedule_protobuf(const std::vector<schedule> &network_schedule, const std::string &schedule_type);

    };

}

#endif //DNNSIM_NETWRITER_H
