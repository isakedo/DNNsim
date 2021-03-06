
#include <core/LocalBuffer.h>

namespace core {

    template <typename T>
    std::string LocalBuffer<T>::header() {
        std::string header = "Number of memory rows: " + std::to_string(ROWS) + "\n";
        if (READ_DELAY != NULL_DELAY) header += "Read delay: " + std::to_string(READ_DELAY) + "\n";
        if (WRITE_DELAY != NULL_DELAY) header += "Write delay: " + std::to_string(WRITE_DELAY) + "\n";
        return header;
    }

    template <typename T>
    void LocalBuffer<T>::configure_layer() {
        size = 0;
        ready_cycle = 0;
    }

    template <typename T>
    void LocalBuffer<T>::insert(bool read) {
        if (read) size++;
        assert(size <= ROWS);
    }

    template <typename T>
    void LocalBuffer<T>::erase(bool read) {
        if (read) {
            assert(size != 0);
            size--;
        }
    }

    template <typename T>
    bool LocalBuffer<T>::isFree() {
        return size < ROWS;
    }

    template <typename T>
    bool LocalBuffer<T>::data_ready() {
        return ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    bool LocalBuffer<T>::write_done() {
        return data_ready();
    }

    template <typename T>
    void LocalBuffer<T>::read_request(bool read) {
        if (read) ready_cycle = *this->global_cycle + READ_DELAY;
    }

    template <typename T>
    void LocalBuffer<T>::write_request(uint64_t delay) {
        ready_cycle = *this->global_cycle + delay + WRITE_DELAY;
    }

    INITIALISE_DATA_TYPES(LocalBuffer);

}
