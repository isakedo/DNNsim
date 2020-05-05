
#include <core/LocalBuffer.h>

namespace core {

    template<typename T>
    uint64_t LocalBuffer<T>::getStallCycles() const {
        return stall_cycles;
    }

    template <typename T>
    void LocalBuffer<T>::configure_layer() {
        idx = 0;
        ready_cycle = std::vector<uint64_t>(ROWS, 0);
        done_cycle = std::vector<uint64_t>(ROWS, 0);

        stall_cycles = 0;
    }

    template <typename T>
    uint64_t LocalBuffer<T>::getFifoReadyCycle() const {
        return ready_cycle[idx];
    }

    template <typename T>
    uint64_t LocalBuffer<T>::getFifoDoneCycle() const {
        return done_cycle[idx];
    }

    template <typename T>
    void LocalBuffer<T>::update_fifo() {
        idx = (idx + 1) % ROWS;
    }

    template <typename T>
    bool LocalBuffer<T>::data_ready() {
        if (ready_cycle[idx] > *this->global_cycle) stall_cycles++;
        return ready_cycle[idx] <= *this->global_cycle;
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
    bool LocalBuffer<T>::write_ready() {
        if (done_cycle[idx] > *this->global_cycle) stall_cycles++;
        return done_cycle[idx] <= *this->global_cycle;
    }

    template <typename T>
    void LocalBuffer<T>::write_request(uint64_t extra_delay) {
        ready_cycle[idx] = *this->global_cycle + WRITE_DELAY + extra_delay;
    }

    template <typename T>
    void LocalBuffer<T>::update_done_cycle(uint64_t global_buffer_ready_cycle) {
        done_cycle[idx] = global_buffer_ready_cycle;
    }


    INITIALISE_DATA_TYPES(LocalBuffer);

}
