
#include <core/GlobalBuffer.h>

namespace core {

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

    template <typename T>
    uint32_t GlobalBuffer<T>::getReadDelay() const {
        return READ_DELAY;
    }

    template <typename T>
    uint32_t GlobalBuffer<T>::getWriteDelay() const {
        return WRITE_DELAY;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActReadReadyCycle() const {
        return act_read_ready_cycle;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtReadReadyCycle() const {
        return wgt_read_ready_cycle;
    }

    template <typename T>
    void GlobalBuffer<T>::configure_layer() {
        act_read_ready_cycle = 0;
        wgt_read_ready_cycle = 0;
        write_ready_cycle = 0;
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

            }

            auto bank_delay = *std::max_element(bank_conflicts.begin(), bank_conflicts.end());

            act_read_ready_cycle = start_time + bank_delay * READ_DELAY;

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

            }

            auto bank_delay = *std::max_element(bank_conflicts.begin(), bank_conflicts.end());

            wgt_read_ready_cycle = start_time + bank_delay * READ_DELAY;

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::write_request(const std::vector<TileData<T>> &tiles_data) {

        auto start_time = std::max(write_ready_cycle, *this->global_cycle);

        auto out_bank_conflicts = std::vector<int>(OUT_BANKS, 0);
        for (const auto &tile_data : tiles_data) {
            for (const auto &out_bank : tile_data.out_banks)
                out_bank_conflicts[out_bank]++;
        }
        auto out_delay = *std::max_element(out_bank_conflicts.begin(), out_bank_conflicts.end()) - 1;
        write_ready_cycle = start_time + out_delay * WRITE_DELAY;

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
