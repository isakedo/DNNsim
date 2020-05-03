
#include <core/Control.h>

namespace core {

    template<typename T>
    uint32_t Control<T>::getActBlks() const {
        return ACT_BLKS;
    }

    template<typename T>
    uint32_t Control<T>::getWgtBlks() const {
        return WGT_BLKS;
    }

    template <typename T>
    const std::shared_ptr<DRAM<T>> &Control<T>::getDram() const {
        return dram;
    }

    template <typename T>
    const std::shared_ptr<GlobalBuffer<T>> &Control<T>::getGbuffer() const {
        return gbuffer;
    }

    template <typename T>
    const std::shared_ptr<LocalBuffer<T>> &Control<T>::getAbuffer() const {
        return abuffer;
    }

    template <typename T>
    const std::shared_ptr<LocalBuffer<T>> &Control<T>::getWbuffer() const {
        return wbuffer;
    }

    template <typename T>
    const std::shared_ptr<LocalBuffer<T>> &Control<T>::getObuffer() const {
        return obuffer;
    }

    template <typename T>
    const std::shared_ptr<Architecture<T>> &Control<T>::getArch() const {
        return arch;
    }

    template <typename T>
    void Control<T>::setArch(const std::shared_ptr<Architecture<T>> &_arch) {
        Control::arch = _arch;
    }

    template <typename T>
    void Control<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
            bool __3dim, int _stride) {

        act = _act;
        wgt = _wgt;
        linear = _linear;
        _3dim = __3dim;

        stride = _stride;

        layer_act_on_chip = next_layer_act_on_chip;
        next_layer_act_on_chip = false;

        ACT_BLKS = (uint32_t) ceil(act_prec / (double) arch->getPeWidth());
        WGT_BLKS = (uint32_t) ceil(wgt_prec / (double) arch->getPeWidth());

        EF_LANES = arch->getLanes();
        EF_COLUMNS = arch->getColumns() / ACT_BLKS;
        EF_ROWS = arch->getRows() / WGT_BLKS;

        auto act_dram_width = std::max(dram->getBaseDataSize(), (uint32_t)pow(2, ceil(log2(act_prec))));
        auto wgt_dram_width = std::max(dram->getBaseDataSize(), (uint32_t)pow(2, ceil(log2(wgt_prec))));

        dram->configure_layer(act_dram_width, wgt_dram_width);
        gbuffer->configure_layer();
        abuffer->configure_layer();
        wbuffer->configure_layer();
        obuffer->configure_layer();
        arch->configure_layer(act_prec, wgt_prec, -1, _linear, EF_COLUMNS);
    }

    template <typename T>
    const std::vector<AddressRange> &Control<T>::getReadActAddresses() const {
        return on_chip_graph.front()->read_act_addresses;
    }

    template <typename T>
    const std::vector<AddressRange> &Control<T>::getReadWgtAddresses() const {
        return on_chip_graph.front()->read_wgt_addresses;
    }

    template <typename T>
    bool Control<T>::getIfEvictAct() const {
        return on_chip_graph.front()->evict_act;
    }

    template <typename T>
    bool Control<T>::getIfEvictWgt() const {
        return on_chip_graph.front()->evict_wgt;
    }

    template <typename T>
    bool Control<T>::still_off_chip_data() {
        on_chip_graph.erase(on_chip_graph.begin());
        return !on_chip_graph.empty();
    }

    template <typename T>
    bool Control<T>::check_if_write_output(std::vector<TileData<T>> &tiles_data) {
        for (const auto &tile_data : tiles_data)
            if (tile_data.write)
                return true;
        return false;
    }

    INITIALISE_DATA_TYPES(Control);

}
