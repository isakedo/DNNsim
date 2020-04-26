
#include <core/DRAM.h>

namespace core {

    template <typename T>
    uint64_t DRAM<T>::getStartActAddress() const {
        return START_ACT_ADDRESS;
    }

    template <typename T>
    uint64_t DRAM<T>::getStartWgtAddress() const {
        return START_WGT_ADDRESS;
    }

    template <typename T>
    uint32_t DRAM<T>::getValuesPerBlock() const {
        return VALUES_PER_BLOCK;
    }

    template <typename T>
    uint32_t DRAM<T>::getDataSize() const {
        return DATA_SIZE;
    }

    template <typename T>
    void DRAM<T>::configure_layer() {
        this->tracked_data->clear();
        *this->act_addresses = std::make_tuple(NULL_ADDR, 0);
        *this->wgt_addresses = std::make_tuple(NULL_ADDR, 0);
    }

    template <typename T>
    void DRAM<T>::cycle() {
        dram_interface->update();
    }

    template <typename T>
    bool DRAM<T>::data_ready(const std::vector<TileData<T>> &tiles_data) {
        return waiting_addresses.empty();
    }

    template <typename T>
    void DRAM<T>::read_request(const std::vector<TileData<T>> &tiles_data) {
        try {
            for (const auto &tile_data : tiles_data) {

                for (const auto &act_addr_row : tile_data.act_addresses)
                    for (const auto &act_addr : act_addr_row)
                        if (act_addr != NULL_ADDR && (*this->tracked_data).at(act_addr) == NULL_TIME)
                            waiting_addresses[act_addr] = nullptr;

                for (const auto &wgt_addr : tile_data.wgt_addresses)
                    if (wgt_addr != NULL_ADDR && (*this->tracked_data).at(wgt_addr) == NULL_TIME)
                        waiting_addresses[wgt_addr] = nullptr;

            }
        } catch (std::exception &exception) {
            throw std::runtime_error("DRAM error: Waiting for a memory address not requested.");
        }
    }


    template <typename T>
    void DRAM<T>::transaction_request(uint64_t address, bool isWrite) {
        if (dram_interface->willAcceptTransaction()) {
            dram_interface->addTransaction(isWrite, address);
        } else {
            request_queue.push(std::make_tuple(address, isWrite));
        }
    }

    template <typename T>
    void DRAM<T>::read_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle) {

        try {
            (*this->tracked_data).at(address) = *this->global_cycle;

            auto it = waiting_addresses.find(address);
            if (it != waiting_addresses.end())
                waiting_addresses.erase(address);

            if (!request_queue.empty()) {
                auto tuple = request_queue.front();
                transaction_request(std::get<0>(tuple), std::get<1>(tuple));
                request_queue.pop();
            }
        } catch (std::exception &exception) {
            throw std::runtime_error("DRAM error: Waiting for a memory address not requested.");
        }
    }

    template <typename T>
    void DRAM<T>::read_data(const std::vector<AddressRange> &act_addresses,
            const std::vector<AddressRange> &wgt_addresses) {

        for (const auto &addr_range : act_addresses) {
            auto start_addr = std::get<0>(addr_range);
            auto end_addr = std::get<1>(addr_range);
            if (start_addr == NULL_ADDR) continue;

            auto &min_addr = std::get<0>(*this->act_addresses);
            auto &max_addr = std::get<1>(*this->act_addresses);

            if (start_addr < min_addr) min_addr = start_addr;
            if (end_addr > max_addr) max_addr = end_addr;

            for (uint64_t addr = start_addr; addr <= end_addr; addr += BLOCK_SIZE) {
                this->tracked_data->insert({addr, NULL_TIME});
                transaction_request(addr, false);
            }
        }

        for (const auto &addr_range : wgt_addresses) {
            auto start_addr = std::get<0>(addr_range);
            auto end_addr = std::get<1>(addr_range);
            if (start_addr == NULL_ADDR) continue;

            auto &min_addr = std::get<0>(*this->wgt_addresses);
            auto &max_addr = std::get<1>(*this->wgt_addresses);

            if (start_addr < min_addr) min_addr = start_addr;
            if (end_addr > max_addr) max_addr = end_addr;

            for (uint64_t addr = start_addr; addr <= end_addr; addr += BLOCK_SIZE) {
                this->tracked_data->insert({addr, NULL_TIME});
                transaction_request(addr, false);
            }
        }

    }

    INITIALISE_DATA_TYPES(DRAM);

}
