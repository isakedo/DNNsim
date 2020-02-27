#ifndef DNNSIM_DRAM_H
#define DNNSIM_DRAM_H

#include "Memory.h"
#include <DRAMSim.h>

namespace core {

    /**
     *
     */
    template <typename T>
    class DRAM : public Memory<T> {

    private:

        const uint32_t START_ACT_ADDRESS;

        const uint32_t START_WGT_ADDRESS;

        const uint32_t VALUES_PER_BLOCK;

        const uint32_t DATA_SIZE;

        /** Memory system */
        DRAMSim::MultiChannelMemorySystem *dram_interface;

        /** Transactions queue */
        std::queue<std::tuple<uint64_t, bool>> queue;

    public:

        DRAM(const std::shared_ptr<std::map<uint64_t, bool>> &_tracked_data, uint32_t _SIZE, uint32_t _DATA_SIZE,
                uint32_t _START_ACT_ADDRESS, uint32_t _START_WGT_ADDRESS, const std::string &_dram_conf,
                const std::string &_system_conf, const std::string &_network) : Memory<T>(_tracked_data),
                DATA_SIZE(_DATA_SIZE), START_ACT_ADDRESS(_START_ACT_ADDRESS), START_WGT_ADDRESS(_START_WGT_ADDRESS),
                VALUES_PER_BLOCK(8 / _DATA_SIZE) {

            dram_interface = DRAMSim::getMemorySystemInstance(_dram_conf, _system_conf, "./DRAMSim2/",
                    "DNNsim_" + _network, _SIZE);

        }

        const uint32_t getStartActAddress() const;

        const uint32_t getStartWgtAddress() const;

        const uint32_t getValuesPerBlock() const;

        const uint32_t getDataSize() const;

        void cycle();

        bool data_ready(const std::vector<TileData<T>> &tiles_data);

        void read_data(const std::vector<AddressRange> &addresses);

    };

}

#endif //DNNSIM_DRAM_H
