
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
    uint32_t DRAM<T>::getBaseValuesPerBlock() const {
        return BASE_VALUES_PER_BLOCK;
    }

    template <typename T>
    uint32_t DRAM<T>::getBaseDataSize() const {
        return BASE_DATA_SIZE;
    }

    template <typename T>
    uint32_t DRAM<T>::getActValuesPerBlock() const {
        return ACT_VALUES_PER_BLOCK;
    }

    template <typename T>
    uint32_t DRAM<T>::getActDataSize() const {
        return ACT_DATA_SIZE;
    }

    template <typename T>
    uint32_t DRAM<T>::getWgtValuesPerBlock() const {
        return WGT_VALUES_PER_BLOCK;
    }

    template <typename T>
    uint32_t DRAM<T>::getWgtDataSize() const {
        return WGT_DATA_SIZE;
    }

    std::string addr_to_hex(uint64_t address, uint32_t SIZE) {
        std::stringstream stream;
        stream << "0x" << std::setfill ('0') << std::setw(ceil(log2(SIZE * pow(2, 20)) / 4.)) << std::hex << address;
        return stream.str();
    }

    template <typename T>
    std::string DRAM<T>::header() {
        std::string header = "Starting activation address: " + addr_to_hex(START_ACT_ADDRESS, SIZE) + "\n";
        header += "Starting weight address: " + addr_to_hex(START_WGT_ADDRESS, SIZE) + "\n";
        return header;
    }

    template <typename T>
    void DRAM<T>::configure_layer(uint32_t _ACT_DATA_SIZE, uint32_t _WGT_DATA_SIZE) {
        ACT_DATA_SIZE = _ACT_DATA_SIZE;
        ACT_VALUES_PER_BLOCK = BLOCK_SIZE / _ACT_DATA_SIZE;

        WGT_DATA_SIZE = _WGT_DATA_SIZE;
        WGT_VALUES_PER_BLOCK = BLOCK_SIZE / _WGT_DATA_SIZE;

        *this->act_addresses = {NULL_ADDR, 0};
        *this->wgt_addresses = {NULL_ADDR, 0};
        this->tracked_data->clear();

        act_reads = 0;
        wgt_reads = 0;
        out_writes = 0;
        stall_cycles = 0;
    }

    template <typename T>
    std::vector<AddressRange> DRAM<T>::compress_addresses(const std::vector<uint64_t> &addresses) {
        auto tmp_addresses = addresses;
        std::sort(tmp_addresses.begin(), tmp_addresses.end());
        auto last = std::unique(tmp_addresses.begin(), tmp_addresses.end());
        tmp_addresses.erase(last, tmp_addresses.end());

        uint64_t prev_addr = tmp_addresses.front();
        auto addr_tuple = std::make_tuple(prev_addr, NULL_ADDR);
        auto compressed_addresses = std::vector<AddressRange>();

        for (int i = 1; i < tmp_addresses.size(); ++i) {
            const auto &addr = tmp_addresses[i];
            if (addr - prev_addr - BLOCK_SIZE != 0) {
                std::get<1>(addr_tuple) = prev_addr;
                compressed_addresses.emplace_back(addr_tuple);
                std::get<0>(addr_tuple) = addr;
            }
            prev_addr = addr;
        }
        std::get<1>(addr_tuple) = prev_addr;
        compressed_addresses.emplace_back(addr_tuple);

        return compressed_addresses;
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
                            waiting_addresses.insert(act_addr);

                for (const auto &wgt_addr : tile_data.wgt_addresses)
                    if (wgt_addr != NULL_ADDR && (*this->tracked_data).at(wgt_addr) == NULL_TIME)
                        waiting_addresses.insert(wgt_addr);

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

    template <typename T>
    void DRAM<T>::write_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle) {
        if (!request_queue.empty()) {
            auto tuple = request_queue.front();
            transaction_request(std::get<0>(tuple), std::get<1>(tuple));
            request_queue.pop();
        }
    }

    template <typename T>
    void DRAM<T>::write_data(const std::vector<AddressRange> &write_addresses) {
        for (const auto &addr_range : write_addresses) {
            auto start_addr = std::get<0>(addr_range);
            auto end_addr = std::get<1>(addr_range);
            if (start_addr == NULL_ADDR) continue;

            for (uint64_t addr = start_addr; addr <= end_addr; addr += BLOCK_SIZE) {
                transaction_request(addr, true);
                out_writes++;
            }
        }
    }


    INITIALISE_DATA_TYPES(DRAM);

}
