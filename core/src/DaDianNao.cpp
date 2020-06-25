
#include <core/DaDianNao.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string DaDianNao<T>::name() {
        return TCL ? "BitTactical" : "DaDianNao";
    }

    /* CYCLES */

    template <typename T>
    bool DaDianNao<T>::diffy() {
        return false;
    }

    template <typename T>
    bool DaDianNao<T>::schedule() {
        return TCL;
    }

    template <typename T>
    void DaDianNao<T>::process_tiles(const TilesData<T> &tiles_data) {

        this->done_cycle = *this->global_cycle + 1;
        this->ready_cycle = *this->global_cycle + 1;
        this->cycles++;

        for (const auto &tile_data : tiles_data.tiles_data) {

            if (!tile_data.valid)
                continue;

            if (this->linear) {
                auto scheduled_pe = tile_data.filters.size() * this->wgt_blks;
                this->scheduled_pe += scheduled_pe;
                this->idle_pe += this->ROWS - scheduled_pe;
            } else {
                auto scheduled_pe = tile_data.windows.size() * this->act_blks
                        * tile_data.filters.size() * this->wgt_blks;
                this->scheduled_pe += scheduled_pe;
                this->idle_pe += this->COLUMNS * this->ROWS - scheduled_pe;
            }
        }

    }

    /* POTENTIALS */

    template <typename T>
    std::string DaDianNao<T>::filename_pot() {
        return "";
    }

    template <typename T>
    std::string DaDianNao<T>::header_pot() {
        return "";
    }

    template <typename T>
    uint16_t DaDianNao<T>::computeBits(T act, T wgt) {

        if (TCL) {
            if(wgt == 0) return 0;
        }

        return this->network_width * this->network_width;
    }

    INITIALISE_DATA_TYPES(DaDianNao);

}