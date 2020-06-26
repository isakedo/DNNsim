
#include <core/GlobalBuffer.h>
#include <cstdint>

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

    template<typename T>
    uint32_t GlobalBuffer<T>::getAddrsPerAccess() const {
        return ADDRS_PER_ACCESS;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActReads() const {
        return act_reads;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getPsumReads() const {
        return psum_reads;
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
    uint64_t GlobalBuffer<T>::getPsumBankConflicts() const {
        return psum_bank_conflicts;
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
    bool GlobalBuffer<T>::data_ready() const {
        return read_ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    std::string GlobalBuffer<T>::filename() {
        return "_AM" + to_mem_string(ACT_SIZE) + "_WM" + to_mem_string(WGT_SIZE);
    }

    template <typename T>
    std::string GlobalBuffer<T>::header() {
        std::string header = "Activations memory size: " + to_mem_string(ACT_SIZE) + "\n";
        header += "Weight memory size: " + to_mem_string(WGT_SIZE) + "\n";
        header += "Number of activation banks: " + std::to_string(ACT_BANKS) + "\n";
        header += "Number of weight banks: " + std::to_string(WGT_BANKS) + "\n";
        header += "Number of output banks: " + std::to_string(OUT_BANKS) + "\n";
        header += "Bank interface width: " + std::to_string(BANK_WIDTH) + "\n";
        header += "Read delay: " + std::to_string(READ_DELAY) + "\n";
        header += "Write delay: " + std::to_string(WRITE_DELAY) + "\n";
        return header;
    }

    template <typename T>
    void GlobalBuffer<T>::configure_layer() {
        psum_read_ready_cycle = 0;
        read_ready_cycle = 0;
        write_ready_cycle = 0;

        act_reads = 0;
        psum_reads = 0;
        wgt_reads = 0;
        out_writes = 0;

        act_bank_conflicts = 0;
        psum_bank_conflicts = 0;
        wgt_bank_conflicts = 0;
        out_bank_conflicts = 0;
    }

    template<typename T>
    bool GlobalBuffer<T>::write_done() {
        return write_ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    void GlobalBuffer<T>::act_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool layer_act_on_chip) {

        try {

            uint64_t start_time = *this->global_cycle;
            auto bank_conflicts = std::vector<int>(ACT_BANKS, 0);

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid)
                    continue;

                // Start time
                if (!layer_act_on_chip)
                    for (const auto &act_addr_row : tile_data.act_addresses)
                        for (const auto &act_addr : act_addr_row)
                            if (act_addr != NULL_ADDR)
                                if (start_time < (*this->tracked_data).at(act_addr))
                                    start_time = (*this->tracked_data).at(act_addr);

                // Bank conflicts
                for (const auto &act_bank : tile_data.act_banks)
                    if (act_bank != -1)
                        bank_conflicts[act_bank]++;

            }

            auto bank_steps = 0;
            for (const auto &reads : bank_conflicts) {
                auto bank_reads = ceil(reads /(double)ADDRS_PER_ACCESS);
                act_reads += bank_reads;
                if (bank_reads > bank_steps)
                    bank_steps = bank_reads;
            }

            auto act_read_ready_cycle = start_time + bank_steps * READ_DELAY;
            act_bank_conflicts += bank_steps > 0 ? bank_steps - 1 : 0;
            read_ready_cycle = std::max(read_ready_cycle, act_read_ready_cycle);

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::psum_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_psum) {

        try {

            bool first = true;
            uint64_t start_time = 0;
            auto bank_conflicts = std::vector<int>(OUT_BANKS, 0);

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid)
                    continue;

                // Read partial sums if needed
                if (tile_data.read_psum) {

                    if (first) {
                        start_time = std::max(*this->global_cycle, write_ready_cycle);
                        start_time = std::max(start_time, *this->global_cycle);
                        first = false;
                    }

                    // Start time
                    for (const auto &out_addr : tile_data.out_addresses)
                        if (out_addr != NULL_ADDR)
                            if (start_time < (*this->tracked_data).at(out_addr))
                                start_time = (*this->tracked_data).at(out_addr);

                    // Bank conflicts
                    for (const auto &out_bank : tile_data.out_banks)
                        if (out_bank != -1)
                            bank_conflicts[out_bank]++;

                }
            }

            auto bank_steps = 0;
            for (const auto &reads : bank_conflicts) {
                auto bank_reads = ceil(reads /(double)ADDRS_PER_ACCESS);
                psum_reads += bank_reads;
                if (bank_reads > bank_steps)
                    bank_steps = bank_reads;
            }

            psum_read_ready_cycle = start_time + bank_steps * READ_DELAY;
            psum_bank_conflicts += bank_steps > 0 ? bank_steps - 1 : 0;
            read_ready_cycle = std::max(read_ready_cycle, psum_read_ready_cycle);
            read_psum = !first;

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::wgt_read_request(const std::shared_ptr<TilesData<T>> &tiles_data) {

        try {

            uint64_t start_time = *this->global_cycle;
            auto bank_conflicts = std::vector<int>(WGT_BANKS, 0);

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid)
                    continue;

                // Start time
                for (const auto &wgt_addr : tile_data.wgt_addresses)
                    if (wgt_addr != NULL_ADDR)
                        if (start_time < (*this->tracked_data).at(wgt_addr))
                            start_time = (*this->tracked_data).at(wgt_addr);

                // Bank conflicts
                for (const auto &wgt_bank : tile_data.wgt_banks)
                    if (wgt_bank != -1)
                        bank_conflicts[wgt_bank]++;

            }

            auto bank_steps = 0;
            for (const auto &reads : bank_conflicts) {
                auto bank_reads = ceil(reads /(double)ADDRS_PER_ACCESS);
                wgt_reads += bank_reads;
                if (bank_reads > bank_steps)
                    bank_steps = bank_reads;
            }

            auto wgt_read_ready_cycle = start_time + bank_steps * READ_DELAY;
            wgt_bank_conflicts += bank_steps > 0 ? bank_steps - 1 : 0;
            read_ready_cycle = std::max(read_ready_cycle, wgt_read_ready_cycle);

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::write_request(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto start_time = std::max(psum_read_ready_cycle, write_ready_cycle);
        start_time = std::max(start_time, *this->global_cycle);

        auto bank_conflicts = std::vector<int>(OUT_BANKS, 0);
        for (const auto &tile_data : tiles_data->data) {

            if (!tile_data.write)
                continue;

            // Bank conflicts
            for (const auto &out_bank : tile_data.out_banks)
                if (out_bank != -1)
                    bank_conflicts[out_bank]++;

        }

        auto bank_steps = 0;
        for (const auto &writes : bank_conflicts) {
            auto bank_writes = ceil(writes /(double)ADDRS_PER_ACCESS);
            out_writes += bank_writes;
            if (bank_writes > bank_steps)
                bank_steps = bank_writes;
        }

        write_ready_cycle = start_time + bank_steps * WRITE_DELAY;
        out_bank_conflicts += bank_steps > 0 ? bank_steps - 1 : 0;

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
