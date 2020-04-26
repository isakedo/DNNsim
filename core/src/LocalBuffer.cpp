
#include <core/LocalBuffer.h>

namespace core {

    template <typename T>
    uint32_t LocalBuffer<T>::getRows() const {
        return ROWS;
    }

    template <typename T>
    uint32_t LocalBuffer<T>::getReadDelay() const {
        return READ_DELAY;
    }

    template <typename T>
    uint32_t LocalBuffer<T>::getWriteDelay() const {
        return WRITE_DELAY;
    }

    template <typename T>
    void LocalBuffer<T>::configure_layer() {
        ready_idx = 0;
        read_ready_cycle = std::vector<uint64_t>(ROWS, 0);
        write_idx = 0;
        write_ready_cycle = std::vector<uint64_t>(ROWS, 0);
    }

    template <typename T>
    bool LocalBuffer<T>::data_ready() {
        return read_ready_cycle[ready_idx] <= *this->global_cycle;
    }

    template <typename T>
    void LocalBuffer<T>::read_request(uint64_t global_buffer_ready_cycle) {

        ready_idx = (ready_idx + 1) % ROWS;
        read_ready_cycle[ready_idx] = global_buffer_ready_cycle + WRITE_DELAY + READ_DELAY;

    }

    INITIALISE_DATA_TYPES(LocalBuffer);

}
