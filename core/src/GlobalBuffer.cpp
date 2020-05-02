
#include <core/GlobalBuffer.h>

namespace core {

    template<typename T>
    uint64_t GlobalBuffer<T>::getActReads() const {
        return act_reads;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtReads() const {
        return wgt_reads;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getOutWrites() const {
        return out_writes;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActBankConflicts() const {
        return act_bank_conflicts;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtBankConflicts() const {
        return wgt_bank_conflicts;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getOutBankConflicts() const {
        return out_bank_conflicts;
    }

    template <typename T>
    uint64_t GlobalBuffer<T>::getActSize() const {
        return ACT_SIZE;
    }

    template <typename T>
    uint64_t GlobalBuffer<T>::getWgtSize() const {
        return WGT_SIZE;
    }

    template <typename T>
    uint32_t GlobalBuffer<T>::getActBanks() const {
        return ACT_BANKS;
    }

    template <typename T>
    uint32_t GlobalBuffer<T>::getWgtBanks() const {
        return WGT_BANKS;
    }

    template<typename T>
    uint32_t GlobalBuffer<T>::getOutBanks() const {
        return OUT_BANKS;
    }

    template <typename T>
    uint32_t GlobalBuffer<T>::getBankWidth() const {
        return BANK_WIDTH;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActReadReadyCycle() const {
        return act_read_ready_cycle;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtReadReadyCycle() const {
        return wgt_read_ready_cycle;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWriteReadyCycle() const {
        return write_ready_cycle;
    }

    template <typename T>
    void GlobalBuffer<T>::configure_layer() {
        act_read_ready_cycle = 0;
        wgt_read_ready_cycle = 0;
        write_ready_cycle = 0;

        act_reads = 0;
        wgt_reads = 0;
        out_writes = 0;
        act_bank_conflicts = 0;
        wgt_bank_conflicts = 0;
        out_bank_conflicts = 0;
    }

    template<typename T>
    bool GlobalBuffer<T>::write_done() {
        return write_ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    void GlobalBuffer<T>::act_read_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle) {

        try {

            uint64_t start_time = std::max(act_read_ready_cycle, fifo_ready_cycle);
            auto bank_conflicts = std::vector<int>(ACT_BANKS, 0);

            for (const auto &tile_data : tiles_data) {

                if (!tile_data.valid)
                    continue;

                // Update start time
                for (const auto &act_addr_row : tile_data.act_addresses)
                    for (const auto &act_addr : act_addr_row)
                        if (act_addr != NULL_ADDR)
                            if (start_time < (*this->tracked_data).at(act_addr))
                                start_time = (*this->tracked_data).at(act_addr);

                // Bank conflicts
                for (const auto &act_bank : tile_data.act_banks)
                    bank_conflicts[act_bank]++;

                if (!tile_data.act_banks.empty())
                    act_reads++;

            }

            auto bank_delay = *std::max_element(bank_conflicts.begin(), bank_conflicts.end());
            act_read_ready_cycle = start_time + bank_delay * READ_DELAY - 1;
            act_bank_conflicts += bank_delay - 1;

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::wgt_read_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle) {

        try {

            uint64_t start_time = std::max(wgt_read_ready_cycle, fifo_ready_cycle);
            auto bank_conflicts = std::vector<int>(WGT_BANKS, 0);

            for (const auto &tile_data : tiles_data) {

                if (!tile_data.valid)
                    continue;

                // Start time
                for (const auto &wgt_addr : tile_data.wgt_addresses)
                    if (wgt_addr != NULL_ADDR)
                        if (start_time < (*this->tracked_data).at(wgt_addr))
                            start_time = (*this->tracked_data).at(wgt_addr);

                // Bank conflicts
                for (const auto &wgt_bank : tile_data.wgt_banks)
                    bank_conflicts[wgt_bank]++;

                if (!tile_data.wgt_banks.empty())
                    wgt_reads++;

            }

            auto bank_delay = *std::max_element(bank_conflicts.begin(), bank_conflicts.end());
            wgt_read_ready_cycle = start_time + bank_delay * READ_DELAY - 1;
            wgt_bank_conflicts += bank_delay - 1;

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::write_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle) {

        auto start_time = std::max(write_ready_cycle, fifo_ready_cycle);

        auto bank_conflicts = std::vector<int>(OUT_BANKS, 0);
        for (const auto &tile_data : tiles_data) {

            if (!tile_data.write)
                continue;

            // Bank conflicts
            for (const auto &out_bank : tile_data.out_banks)
                bank_conflicts[out_bank]++;

            if (!tile_data.out_banks.empty())
                out_writes++;

        }

        auto bank_delay = *std::max_element(bank_conflicts.begin(), bank_conflicts.end());
        write_ready_cycle = start_time + bank_delay * WRITE_DELAY - 1;
        out_bank_conflicts += bank_delay - 1;

    }

    template <typename T>
    void GlobalBuffer<T>::evict_data(bool evict_act, bool evict_wgt) {
        if (evict_act) {

            auto min_addr = std::get<0>(*this->act_addresses);
            auto max_addr = std::get<1>(*this->act_addresses);

            if (min_addr != NULL_ADDR) {
                auto it = this->tracked_data->find(min_addr);
                auto it2 = this->tracked_data->find(max_addr);
                this->tracked_data->erase(it, it2);
                this->tracked_data->erase(max_addr);
                *this->act_addresses = {NULL_ADDR, 0};
            }

        }

        if (evict_wgt) {

            auto min_addr = std::get<0>(*this->wgt_addresses);
            auto max_addr = std::get<1>(*this->wgt_addresses);

            if (min_addr != NULL_ADDR) {
                auto it = this->tracked_data->find(min_addr);
                auto it2 = this->tracked_data->find(max_addr);
                this->tracked_data->erase(it, it2);
                this->tracked_data->erase(max_addr);
                *this->wgt_addresses = {NULL_ADDR, 0};
            }

        }
    }

    INITIALISE_DATA_TYPES(GlobalBuffer);

}
