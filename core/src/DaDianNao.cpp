
#include <core/DaDianNao.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string DaDianNao<T>::name() {
        return TCL ? "BitTactical" : "DaDianNao";
    }

    /* CYCLES */

    template <typename T>
    std::string DaDianNao<T>::filename() {
        return "_L" + std::to_string(this->N_LANES) + "_C" + std::to_string(this->N_COLUMNS) +
               "_R" + std::to_string(this->N_ROWS) + "_T" + std::to_string(this->N_TILES) +
               "_BP" + std::to_string(this->BITS_PE);
    }

    template <typename T>
    std::string DaDianNao<T>::header() {
        std::string header = "Number of lanes/terms per PE: " + std::to_string(this->N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(this->N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(this->N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(this->N_TILES) + "\n";
        header += "Size of the PE in bits: " + std::to_string(this->BITS_PE) + "\n";
        return header;
    }

    template <typename T>
    bool DaDianNao<T>::diffy() {
        return false;
    }

    template <typename T>
    bool DaDianNao<T>::schedule() {
        return TCL;
    }

    template <typename T>
    void DaDianNao<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {

        this->done_cycle = *this->global_cycle + 1;
        this->ready_cycle = *this->global_cycle + 1;
        this->cycles++;

        for (const auto &tile_data : tiles_data) {

            if (!tile_data.valid)
                continue;

            auto ROWS = tile_data.wgt_row.size() / tile_data.lanes;
            auto COLUMNS = tile_data.act_row.front().size() / tile_data.lanes;
            if (this->linear) {
                this->scheduled_pe += tile_data.filters.size();
                this->idle_pe += ROWS - tile_data.filters.size();
            } else {
                this->scheduled_pe += tile_data.windows.size() * tile_data.filters.size();
                this->idle_pe += (COLUMNS * ROWS - tile_data.windows.size() * tile_data.filters.size());
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