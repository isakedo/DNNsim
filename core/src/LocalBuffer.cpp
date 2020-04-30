
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
        idx = 0;
        ready_cycle = std::vector<uint64_t>(ROWS, 0);
        done_cycle = std::vector<uint64_t>(ROWS, 0);
    }

    template <typename T>
    bool LocalBuffer<T>::data_ready() {
        return ready_cycle[idx] <= *this->global_cycle;
    }

    template <typename T>
    uint64_t LocalBuffer<T>::getFifoReadyCycle() {
        return done_cycle[idx];
    }

    template <typename T>
    void LocalBuffer<T>::read_request(uint64_t global_buffer_ready_cycle) {
        ready_cycle[idx] = global_buffer_ready_cycle + READ_DELAY;
    }

    template <typename T>
    void LocalBuffer<T>::evict_data() {
        done_cycle[idx] = *this->global_cycle;
    }

    template <typename T>
    void LocalBuffer<T>::update_fifo() {
        idx = (idx + 1) % ROWS;
    }

    INITIALISE_DATA_TYPES(LocalBuffer);

}
