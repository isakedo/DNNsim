
#include <core/GlobalBuffer.h>
#include <cstdint>

namespace core {

    template <typename T>
    uint64_t GlobalBuffer<T>::getActLevels() const {
        return ACT_LEVELS;
    }

    template <typename T>
    uint64_t GlobalBuffer<T>::getWgtLevels() const {
        return WGT_LEVELS;
    }

    template <typename T>
    uint64_t GlobalBuffer<T>::getActSize() const {
        return ACT_SIZE.front();
    }

    template <typename T>
    uint64_t GlobalBuffer<T>::getWgtSize() const {
        return WGT_SIZE.front();
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
    uint32_t GlobalBuffer<T>::getActAddrsPerAccess() const {
        return ACT_ADDRS_PER_ACCESS;
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActReads(uint32_t idx) const {
        return act_reads[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getPsumReads(uint32_t idx) const {
        return psum_reads[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtReads(uint32_t idx) const {
        return wgt_reads[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getOutWrites(uint32_t idx) const {
        return out_writes[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getActBankConflicts(uint32_t idx) const {
        return act_bank_conflicts[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getPsumBankConflicts(uint32_t idx) const {
        return psum_bank_conflicts[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getWgtBankConflicts(uint32_t idx) const {
        return wgt_bank_conflicts[idx];
    }

    template<typename T>
    uint64_t GlobalBuffer<T>::getOutBankConflicts(uint32_t idx) const {
        return out_bank_conflicts[idx];
    }

    template <typename T>
    bool GlobalBuffer<T>::data_ready() const {
        return read_ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    std::string GlobalBuffer<T>::filename() {
        return "_AM" + to_mem_string(ACT_SIZE.front()) + "_WM" + to_mem_string(WGT_SIZE.front());
    }

    template <typename T>
    std::string GlobalBuffer<T>::header() {
        std::string header = "Activations memory size: ";
        for (const auto &size : ACT_SIZE) header += to_mem_string(size) + " "; header += "\n";
        header += "Weight memory size: ";
        for (const auto &size : WGT_SIZE) header += to_mem_string(size) + " "; header += "\n";
        header += "Number of activation banks: " + std::to_string(ACT_BANKS) + "\n";
        header += "Number of weight banks: " + std::to_string(WGT_BANKS) + "\n";
        header += "Number of output banks: " + std::to_string(OUT_BANKS) + "\n";
        header += "Activation bank interface width: " + std::to_string(ACT_BANK_WIDTH) + "\n";
        header += "Weight bank interface width: " + std::to_string(WGT_BANK_WIDTH) + "\n";
        header += "Activations read delay: ";
        for (const auto &delay : ACT_READ_DELAY) header += std::to_string(delay) + " "; header += "\n";
        header += "Activations write delay: ";
        for (const auto &delay : ACT_WRITE_DELAY) header += std::to_string(delay) + " "; header += "\n";
        header += "Weights read delay: ";
        for (const auto &delay : WGT_READ_DELAY) header += std::to_string(delay) + " "; header += "\n";
        header += "Activations eviction policy: " + ACT_POLICY + "\n";
        header += "Weights eviction policy: " + WGT_POLICY + "\n";
        return header;
    }

    template <typename T>
    void GlobalBuffer<T>::configure_layer() {
        psum_read_ready_cycle = 0;
        read_ready_cycle = 0;
        write_ready_cycle = 0;

        act_reads = std::vector<uint64_t>(ACT_LEVELS, 0);
        psum_reads = std::vector<uint64_t>(ACT_LEVELS, 0);
        wgt_reads = std::vector<uint64_t>(WGT_LEVELS, 0);
        out_writes = std::vector<uint64_t>(ACT_LEVELS, 0);

        act_bank_conflicts = std::vector<uint64_t>(ACT_LEVELS, 0);
        psum_bank_conflicts = std::vector<uint64_t>(ACT_LEVELS, 0);
        wgt_bank_conflicts = std::vector<uint64_t>(WGT_LEVELS, 0);
        out_bank_conflicts = std::vector<uint64_t>(ACT_LEVELS, 0);

        for (int lvl = 1; lvl < ACT_LEVELS; ++lvl) {
            for (int bank = 0; bank < ACT_BANKS; ++bank) {
                act_eviction_policy[lvl][bank]->flush();
            }
            for (int bank = 0; bank < OUT_BANKS; ++bank) {
                out_eviction_policy[lvl][bank]->flush();
            }
        }

        for (int lvl = 1; lvl < WGT_LEVELS; ++lvl) {
            for (int bank = 0; bank < WGT_BANKS; ++bank) {
                wgt_eviction_policy[lvl][bank]->flush();
            }
        }
    }

    template<typename T>
    bool GlobalBuffer<T>::write_done() {
        return write_ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    void GlobalBuffer<T>::act_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool layer_act_on_chip,
            bool &read_act) {

        try {

            auto bank_addr_reads = std::vector<std::vector<int>>(ACT_LEVELS, std::vector<int>(ACT_BANKS, 0));

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid || tile_data.act_addresses.empty())
                    continue;

                assert(tile_data.act_banks.size() == tile_data.act_addresses.size());
                assert(tile_data.act_banks.front().size() == tile_data.act_addresses.front().size());

                uint64_t rows = tile_data.act_banks.size();
                uint64_t n_addr = tile_data.act_banks.front().size();
                for (int row = 0; row < rows; ++row) {
                    for (int idx = 0; idx < n_addr; ++idx) {

                        const auto &act_addr = tile_data.act_addresses[row][idx];
                        if (act_addr == NULL_ADDR)
                            continue;

                        if (layer_act_on_chip) {
                            auto it = (*this->tracked_data).find(act_addr);
                            if (it == (*this->tracked_data).end())
                                this->tracked_data->insert({act_addr,  1});
                        }

                        read_act = true;
                        const auto &act_lvl = (*this->tracked_data).at(act_addr);
                        const auto &act_bank = tile_data.act_banks[row][idx];

                        assert(act_bank != -1);
                        assert(act_lvl >= 1 && act_lvl <= ACT_LEVELS);

                        for (int lvl = ACT_LEVELS; lvl > act_lvl; --lvl) {
                            if (!act_eviction_policy[lvl - 1][act_bank]->free_entry()) {
                                auto evict_addr = act_eviction_policy[lvl - 1][act_bank]->evict_addr();
                                assert((*this->tracked_data).at(evict_addr) == lvl);
                                (*this->tracked_data).at(evict_addr) = lvl - 1;
                            }
                            act_eviction_policy[lvl - 1][act_bank]->insert_addr(act_addr);
                        }

                        for (int lvl = ACT_LEVELS; lvl >= 1; --lvl) {
                            if (lvl >= act_lvl) bank_addr_reads[lvl - 1][act_bank]++;
                            else if (lvl != 1) act_eviction_policy[lvl - 1][act_bank]->update_status(act_addr);
                        }

                        (*this->tracked_data).at(act_addr) = ACT_LEVELS;

                    }
                }

            }

            uint64_t start_time = read_act ? *this->global_cycle : 0;
            for (int lvl = 0; lvl < ACT_LEVELS; ++lvl) {

                auto bank_steps = 0;
                for (const auto &reads : bank_addr_reads[lvl]) {
                    auto bank_reads = ceil(reads / (double)ACT_ADDRS_PER_ACCESS);
                    act_reads[lvl] += bank_reads;
                    if (bank_reads > bank_steps)
                        bank_steps = bank_reads;
                }

                start_time += bank_steps * ACT_READ_DELAY[lvl];
                act_bank_conflicts[lvl] += bank_steps > 0 ? bank_steps - 1 : 0;

            }

            read_ready_cycle = std::max(read_ready_cycle, start_time);

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::psum_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_psum) {

        try {

            auto bank_addr_reads = std::vector<std::vector<int>>(ACT_LEVELS, std::vector<int>(OUT_BANKS, 0));

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid || tile_data.psum_addresses.empty())
                    continue;

                assert(tile_data.psum_banks.size() == tile_data.psum_addresses.size());

                uint64_t n_addr = tile_data.psum_banks.size();
                for (int idx = 0; idx < n_addr; ++idx) {

                    const auto &psum_addr = tile_data.psum_addresses[idx];
                    if (psum_addr == NULL_ADDR)
                        continue;

                    read_psum = true;
                    const auto &psum_lvl = (*this->tracked_data).at(psum_addr);
                    const auto &psum_bank = tile_data.psum_banks[idx];

                    assert(psum_bank != -1);
                    assert(psum_lvl >= 1 && psum_lvl <= ACT_LEVELS);

                    for (int lvl = ACT_LEVELS; lvl > psum_lvl; --lvl) {
                        if (!out_eviction_policy[lvl - 1][psum_bank]->free_entry()) {
                            auto evict_addr = out_eviction_policy[lvl - 1][psum_bank]->evict_addr();
                            assert((*this->tracked_data).at(evict_addr) == lvl);
                            (*this->tracked_data).at(evict_addr) = lvl - 1;
                        }
                        out_eviction_policy[lvl - 1][psum_bank]->insert_addr(psum_addr);
                    }

                    for (int lvl = ACT_LEVELS; lvl >= 1; --lvl) {
                        if (lvl >= psum_lvl) bank_addr_reads[lvl - 1][psum_bank]++;
                        else if (lvl != 1) out_eviction_policy[lvl - 1][psum_bank]->update_status(psum_addr);
                    }

                    (*this->tracked_data).at(psum_addr) = ACT_LEVELS;

                }

            }

            uint64_t start_time = read_psum ? std::max(*this->global_cycle, write_ready_cycle) : 0;
            for (int lvl = 0; lvl < ACT_LEVELS; ++lvl) {

                auto bank_steps = 0;
                for (const auto &reads : bank_addr_reads[lvl]) {
                    auto bank_reads = ceil(reads / (double)ACT_ADDRS_PER_ACCESS);
                    psum_reads[lvl] += bank_reads;
                    if (bank_reads > bank_steps)
                        bank_steps = bank_reads;
                }

                start_time += bank_steps * ACT_READ_DELAY[lvl];
                psum_bank_conflicts[lvl] += bank_steps > 0 ? bank_steps - 1 : 0;

            }

            psum_read_ready_cycle = start_time;
            read_ready_cycle = std::max(read_ready_cycle, psum_read_ready_cycle);

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::wgt_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_wgt) {

        try {

            auto bank_addr_reads = std::vector<std::vector<int>>(WGT_LEVELS, std::vector<int>(WGT_BANKS, 0));

            for (const auto &tile_data : tiles_data->data) {

                if (!tile_data.valid || tile_data.wgt_addresses.empty())
                    continue;

                assert(tile_data.wgt_banks.size() == tile_data.wgt_addresses.size());

                uint64_t n_addr = tile_data.wgt_banks.size();
                for (int idx = 0; idx < n_addr; ++idx) {

                    const auto &wgt_addr = tile_data.wgt_addresses[idx];
                    if (wgt_addr == NULL_ADDR)
                        continue;

                    read_wgt = true;
                    const auto &wgt_lvl = (*this->tracked_data).at(wgt_addr);
                    const auto &wgt_bank = tile_data.wgt_banks[idx];

                    assert(wgt_bank != -1);
                    assert(wgt_lvl >= 1 && wgt_lvl <= WGT_LEVELS);

                    for (int lvl = WGT_LEVELS; lvl > wgt_lvl; --lvl) {
                        if (!wgt_eviction_policy[lvl - 1][wgt_bank]->free_entry()) {
                            auto evict_addr = wgt_eviction_policy[lvl - 1][wgt_bank]->evict_addr();
                            assert((*this->tracked_data).at(evict_addr) == lvl);
                            (*this->tracked_data).at(evict_addr) = lvl - 1;
                        }
                        wgt_eviction_policy[lvl - 1][wgt_bank]->insert_addr(wgt_addr);
                    }

                    for (int lvl = WGT_LEVELS; lvl >= 1; --lvl) {
                        if (lvl >= wgt_lvl) bank_addr_reads[lvl - 1][wgt_bank]++;
                        else if (lvl != 1) act_eviction_policy[lvl - 1][wgt_bank]->update_status(wgt_addr);
                    }

                    (*this->tracked_data).at(wgt_addr) = WGT_LEVELS;

                }

            }

            uint64_t start_time = read_wgt ? *this->global_cycle : 0;
            for (int lvl = 0; lvl < WGT_LEVELS; ++lvl) {

                auto bank_steps = 0;
                for (const auto &reads : bank_addr_reads[lvl]) {
                    auto bank_reads = ceil(reads / (double)WGT_ADDRS_PER_ACCESS);
                    wgt_reads[lvl] += bank_reads;
                    if (bank_reads > bank_steps)
                        bank_steps = bank_reads;
                }

                start_time += bank_steps * WGT_READ_DELAY[lvl];
                wgt_bank_conflicts[lvl] += bank_steps > 0 ? bank_steps - 1 : 0;

            }

            read_ready_cycle = std::max(read_ready_cycle, start_time);

        } catch (std::exception &exception) {
            throw std::runtime_error("Global Buffer waiting for a memory address not requested.");
        }

    }

    template <typename T>
    void GlobalBuffer<T>::write_request(const std::shared_ptr<TilesData<T>> &tiles_data) {

        auto bank_addr_writes = std::vector<std::vector<int>>(ACT_LEVELS, std::vector<int>(OUT_BANKS, 0));

        for (const auto &tile_data : tiles_data->data) {

            if (!tile_data.valid || tile_data.out_addresses.empty())
                continue;

            assert(tile_data.out_banks.size() == tile_data.out_addresses.size());

            uint64_t n_addr = tile_data.out_banks.size();
            for (int idx = 0; idx < n_addr; ++idx) {

                const auto &out_addr = tile_data.out_addresses[idx];
                if (out_addr == NULL_ADDR)
                    continue;

                auto it = (*this->tracked_data).find(out_addr);
                if (it == (*this->tracked_data).end()) {
                    this->tracked_data->insert({out_addr, 1});

                    auto &min_addr = std::get<0>(*this->out_addresses);
                    auto &max_addr = std::get<1>(*this->out_addresses);

                    if (out_addr < min_addr) min_addr = out_addr;
                    if (out_addr > max_addr) max_addr = out_addr;
                }

                const auto &out_lvl = (*this->tracked_data).at(out_addr);
                const auto &out_bank = tile_data.out_banks[idx];

                assert(out_bank != -1);
                assert(out_lvl >= 1 && out_lvl <= ACT_LEVELS);

                for (int lvl = ACT_LEVELS; lvl >= out_lvl; --lvl) {
                    bank_addr_writes[lvl - 1][out_bank]++;

                    if (lvl != 1 && out_lvl != ACT_LEVELS) {
                        if (!out_eviction_policy[lvl - 1][out_bank]->free_entry()) {
                            auto evict_addr = out_eviction_policy[lvl - 1][out_bank]->evict_addr();
                            assert((*this->tracked_data).at(evict_addr) == lvl);
                            (*this->tracked_data).at(evict_addr) = lvl - 1;
                        }
                        out_eviction_policy[lvl - 1][out_bank]->insert_addr(out_addr);
                    }
                }

                (*this->tracked_data).at(out_addr) = ACT_LEVELS;

            }

        }

        uint64_t start_time = std::max(*this->global_cycle, psum_read_ready_cycle);
        for (int lvl = 0; lvl < ACT_LEVELS; ++lvl) {

            auto bank_steps = 0;
            for (const auto &writes : bank_addr_writes[lvl]) {
                auto bank_writes = ceil(writes / (double)ACT_ADDRS_PER_ACCESS);
                out_writes[lvl] += bank_writes;
                if (bank_writes > bank_steps)
                    bank_steps = bank_writes;
            }

            start_time += bank_steps * ACT_WRITE_DELAY[lvl];
            out_bank_conflicts[lvl] += bank_steps > 0 ? bank_steps - 1 : 0;

        }

        write_ready_cycle = start_time;

    }

    template <typename T>
    void GlobalBuffer<T>::evict_data(bool evict_act, bool evict_out, bool evict_wgt) {
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

            for (int lvl = 1; lvl < ACT_LEVELS; ++lvl) {
                for (int bank = 0; bank < ACT_BANKS; ++bank) {
                    act_eviction_policy[lvl][bank]->flush();
                }
            }

        }

        if (evict_out) {

            auto min_addr = std::get<0>(*this->out_addresses);
            auto max_addr = std::get<1>(*this->out_addresses);

            if (min_addr != NULL_ADDR) {
                auto it = this->tracked_data->find(min_addr);
                auto it2 = this->tracked_data->find(max_addr);
                this->tracked_data->erase(it, it2);
                this->tracked_data->erase(max_addr);
                *this->out_addresses = {NULL_ADDR, 0};
            }

            for (int lvl = 1; lvl < ACT_LEVELS; ++lvl) {
                for (int bank = 0; bank < OUT_BANKS; ++bank) {
                    out_eviction_policy[lvl][bank]->flush();
                }
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

            for (int lvl = 1; lvl < WGT_LEVELS; ++lvl) {
                for (int bank = 0; bank < WGT_BANKS; ++bank) {
                    wgt_eviction_policy[lvl][bank]->flush();
                }
            }

        }
    }

    INITIALISE_DATA_TYPES(GlobalBuffer);

}
