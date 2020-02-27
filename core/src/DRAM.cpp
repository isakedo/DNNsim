
#include <core/DRAM.h>

namespace core {

    template <typename T>
    void DRAM<T>::cycle() {
        dram_interface->update();
    }

    template <typename T>
    bool DRAM<T>::data_ready(const std::vector<TileData<T>> &tiles_data) {

    }

    template <typename T>
    void DRAM<T>::read_data(const std::vector<AddressRange> &addresses) {

    }

    template <typename T>
    const uint32_t DRAM<T>::getStartActAddress() const {
        return START_ACT_ADDRESS;
    }

    template <typename T>
    const uint32_t DRAM<T>::getStartWgtAddress() const {
        return START_WGT_ADDRESS;
    }

    template <typename T>
    const uint32_t DRAM<T>::getValuesPerBlock() const {
        return VALUES_PER_BLOCK;
    }

    template <typename T>
    const uint32_t DRAM<T>::getDataSize() const {
        return DATA_SIZE;
    }

    INITIALISE_DATA_TYPES(DRAM);

}
