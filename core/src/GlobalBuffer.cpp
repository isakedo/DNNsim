
#include <core/GlobalBuffer.h>

namespace core {

    template <typename T>
    const uint64_t GlobalBuffer<T>::getSize() const {
        return SIZE;
    }

    template <typename T>
    const uint32_t GlobalBuffer<T>::getActBanks() const {
        return ACT_BANKS;
    }

    template <typename T>
    const uint32_t GlobalBuffer<T>::getWgtBanks() const {
        return WGT_BANKS;
    }

    template <typename T>
    const uint32_t GlobalBuffer<T>::getBankWidth() const {
        return BANK_WIDTH;
    }

    template <typename T>
    const uint32_t GlobalBuffer<T>::getReadDelay() const {
        return READ_DELAY;
    }

    template <typename T>
    const uint32_t GlobalBuffer<T>::getWriteDelay() const {
        return WRITE_DELAY;
    }

    template <typename T>
    void GlobalBuffer<T>::read_request(const std::vector<TileData<T>> &tiles_data) {

        auto start_time = std::max(ready_cycle, *this->global_cycle);

        // Calculate bank conflicts
        auto act_bank_conflicts = std::vector<int>(ACT_BANKS, 0);
        auto wgt_bank_conflicts = std::vector<int>(WGT_BANKS, 0);
        for (const auto &tile_data : tiles_data) {
            for (const auto &act_bank : tile_data.act_banks) act_bank_conflicts[act_bank]++;
            for (const auto &wgt_bank : tile_data.wgt_banks) wgt_bank_conflicts[wgt_bank]++;
        }
        auto act_delay = *std::max_element(act_bank_conflicts.begin(), act_bank_conflicts.end()) - 1;
        auto wgt_delay = *std::max_element(wgt_bank_conflicts.begin(), wgt_bank_conflicts.end()) - 1;
        auto bank_delay = std::max(act_delay, wgt_delay);

        ready_cycle = start_time + bank_delay + READ_DELAY;

    }

    template <typename T>
    void GlobalBuffer<T>::evict_data(const std::vector<AddressRange> &addresses) {
        for (const auto &addr_range : addresses) {
            auto start_addr = std::get<0>(addr_range);
            auto end_addr = std::get<1>(addr_range);
            for (uint64_t addr = start_addr; addr <= end_addr; addr += 0x40) {
                this->tracked_data->erase(addr);
            }
        }
    }

    INITIALISE_DATA_TYPES(GlobalBuffer);

}
