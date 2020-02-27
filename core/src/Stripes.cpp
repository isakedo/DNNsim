
#include <core/Stripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint64_t Stripes<T>::getCycles() const {
        return this->linear ? sys::get_max(this->compute_cycles) : this->cycles;
    }

    template <typename T>
    std::string Stripes<T>::name() {
        return "Stripes";
    }

    /* CYCLES */

    template <typename T>
    std::string Stripes<T>::filename() {
        return "_L" + std::to_string(this->N_LANES) + "_C" + std::to_string(this->N_COLUMNS) +
               "_R" + std::to_string(this->N_ROWS) + "_T" + std::to_string(this->N_TILES) +
               "_BP" + std::to_string(this->BITS_PE);
    }

    template <typename T>
    std::string Stripes<T>::header() {
        std::string header = "Number of lanes/terms per PE: " + std::to_string(this->N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(this->N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(this->N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(this->N_TILES) + "\n";
        header += "Size of the PE in bits: " + std::to_string(this->BITS_PE) + "\n";
        return header;
    }

    template <typename T>
    bool Stripes<T>::diffy() {
        return false;
    }

    template <typename T>
    bool Stripes<T>::schedule() {
        return false;
    }

    template <typename T>
    void Stripes<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {

        if (this->linear) {

            if (this->cycles < this->compute_cycles[this->column_index])
                this->cycles = this->compute_cycles[this->column_index];

            this->compute_cycles[this->column_index] = this->cycles + this->act_prec;
            this->cycles++;

            this->column_cycles[this->column_index] = *this->global_cycle + this->act_prec;
            this->column_index = (this->column_index + 1) % this->column_cycles.size();

            this->done_cycle = *this->global_cycle + this->act_prec;
            this->ready_cycle = this->column_cycles[this->column_index];

        } else {

            this->done_cycle = *this->global_cycle + this->act_prec;
            this->ready_cycle = *this->global_cycle + this->act_prec;
            this->cycles += this->act_prec;

        }

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

    template <typename T>
    bool Stripes<T>::ready() {
        if(this->ready_cycle > *this->global_cycle) this->stall_cycles++;
        return this->ready_cycle <= *this->global_cycle;
    }

    template <typename T>
    bool Stripes<T>::flush() {
        if(this->ready_cycle > *this->global_cycle) this->stall_cycles++;
        return this->done_cycle <= *this->global_cycle;
    }

    /* POTENTIALS */

    template <typename T>
    std::string Stripes<T>::filename_pot() {
        return "";
    }

    template <typename T>
    std::string Stripes<T>::header_pot() {
        return "";
    }

    template <typename T>
    uint16_t Stripes<T>::computeBits(T act, T wgt) {
        return this->act_prec * this->network_bits;
    }

    template class Stripes<uint16_t>;

}
