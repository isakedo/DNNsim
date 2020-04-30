
#include <core/DRAM.h>

namespace core {

    template<typename T>
    uint64_t DRAM<T>::getActReads() const {
        return act_reads;
    }

    template<typename T>
    uint64_t DRAM<T>::getWgtReads() const {
        return wgt_reads;
    }

    template<typename T>
    uint64_t DRAM<T>::getOutWrites() const {
        return out_writes;
    }

    template<typename T>
    uint64_t DRAM<T>::getStallCycles() const {
        return stall_cycles;
    }

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
        *this->act_addresses = {NULL_ADDR, 0};
        *this->wgt_addresses = {NULL_ADDR, 0};
        this->tracked_data->clear();
        act_reads = 0;
        wgt_reads = 0;
        out_writes = 0;
        stall_cycles = 0;
    }

    template <typename T>
    void DRAM<T>::cycle() {
        dram_interface->update();
    }

    template <typename T>
    bool DRAM<T>::data_ready(const std::vector<TileData<T>> &tiles_data) {
        if (!waiting_addresses.empty()) stall_cycles++;
        return waiting_addresses.empty();
    }

    template <typename T>
    void DRAM<T>::read_request(const std::vector<TileData<T>> &tiles_data) {
        try {
            for (const auto &tile_data : tiles_data) {

                if (!tile_data.valid)
                    continue;

                for (const auto &act_addr_row : tile_data.act_addresses)
                    for (const auto &act_addr : act_addr_row)
                        if (act_addr != NULL_ADDR && (*this->tracked_data).at(act_addr) == NULL_TIME)
                            waiting_addresses[act_addr] = nullptr;

                for (const auto &wgt_addr : tile_data.wgt_addresses)
                    if (wgt_addr != NULL_ADDR && (*this->tracked_data).at(wgt_addr) == NULL_TIME)
                        waiting_addresses[wgt_addr] = nullptr;

            }
        } catch (std::exception &exception) {
            throw std::runtime_error("DRAM waiting for a memory address not requested.");
        }
    }


    template <typename T>
    void DRAM<T>::transaction_request(uint64_t address, bool isWrite) {
        if (dram_interface->willAcceptTransaction()) {
            dram_interface->addTransaction(isWrite, address);
        } else {
            request_queue.push({address, isWrite});
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
            throw std::runtime_error("DRAM waiting for a memory address not requested.");
        }
    }

    template <typename T>
    void DRAM<T>::read_data(const std::vector<AddressRange> &act_addresses,
            const std::vector<AddressRange> &wgt_addresses) {

        uint32_t OVERLAP = 16;
        uint32_t act_addr_idx = 0;
        uint64_t act_start_addr = NULL_ADDR;
        uint64_t act_end_addr = 0;
        bool act_first = true;

        uint64_t wgt_addr_idx = 0;
        uint64_t wgt_start_addr = NULL_ADDR;
        uint64_t wgt_end_addr = 0;
        bool wgt_first = true;

        int count = 0;
        bool still_data = true;
        while (still_data) {

            still_data = false;

            count = 0;
            while (act_addr_idx < act_addresses.size()) {

                if (act_first) {
                    const auto &addr_range = act_addresses[act_addr_idx];
                    act_start_addr = std::get<0>(addr_range);
                    act_end_addr = std::get<1>(addr_range);
                    if (act_start_addr == NULL_ADDR) {
                        act_addr_idx++;
                        act_first = true;
                        continue;
                    }

                    auto &min_addr = std::get<0>(*this->act_addresses);
                    auto &max_addr = std::get<1>(*this->act_addresses);

                    if (act_start_addr < min_addr) min_addr = act_start_addr;
                    if (act_end_addr > max_addr) max_addr = act_end_addr;

                    act_first = false;
                }

                while (act_start_addr <= act_end_addr) {
                    if (count == OVERLAP)
                        break;

                    this->tracked_data->insert({act_start_addr, NULL_TIME});
                    transaction_request(act_start_addr, false);
                    still_data = true;
                    act_reads++;

                    act_start_addr += BLOCK_SIZE;
                    count++;
                }

                if (act_start_addr > act_end_addr) {
                    act_first = true;
                    act_addr_idx++;
                }

                break;
            }

            count = 0;
            while (wgt_addr_idx < wgt_addresses.size()) {

                if (wgt_first) {
                    const auto &addr_range = wgt_addresses[wgt_addr_idx];
                    wgt_start_addr = std::get<0>(addr_range);
                    wgt_end_addr = std::get<1>(addr_range);
                    if (wgt_start_addr == NULL_ADDR) {
                        wgt_addr_idx++;
                        wgt_first = true;
                        continue;
                    }

                    auto &min_addr = std::get<0>(*this->wgt_addresses);
                    auto &max_addr = std::get<1>(*this->wgt_addresses);

                    if (wgt_start_addr < min_addr) min_addr = wgt_start_addr;
                    if (wgt_end_addr > max_addr) max_addr = wgt_end_addr;

                    wgt_first = false;
                }

                while (wgt_start_addr <= wgt_end_addr) {

                    if (count == OVERLAP)
                        break;

                    this->tracked_data->insert({wgt_start_addr, NULL_TIME});
                    transaction_request(wgt_start_addr, false);
                    still_data = true;
                    wgt_reads++;

                    wgt_start_addr += BLOCK_SIZE;
                    count++;
                }

                if (wgt_start_addr > wgt_end_addr) {
                    wgt_first = true;
                    wgt_addr_idx++;
                }

                break;
            }

        }

    }

    INITIALISE_DATA_TYPES(DRAM);

}
