
#include <core/Control.h>

namespace core {

    template<typename T>
    uint64_t Control<T>::getCycles() const {
        return *global_cycle;
    }

    template <typename T>
    void Control<T>::cycle() {
        dram->cycle();
        *global_cycle += 1;
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
    const std::shared_ptr<LocalBuffer<T>> &Control<T>::getPbuffer() const {
        return pbuffer;
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
    const std::shared_ptr<Composer<T>> &Control<T>::getComposer() const {
        return composer;
    }

    template <typename T>
    const std::shared_ptr<PPU<T>> &Control<T>::getPPU() const {
        return ppu;
    }

    template <typename T>
    const std::shared_ptr<Architecture<T>> &Control<T>::getArch() const {
        return arch;
    }

    template <typename T>
    void Control<T>::setArch(const std::shared_ptr<Architecture<T>> &_arch) {
        Control::arch = _arch;
        arch->setGlobalCycle(global_cycle);
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

        if (EF_COLUMNS == 0)
            throw std::runtime_error ("Too few columns to perform spatial decomposition");
        if (EF_ROWS == 0)
            throw std::runtime_error ("Too few rows to perform spatial decomposition");

        auto act_dram_width = std::max(dram->getBaseDataSize(), (uint32_t)pow(2, ceil(log2(act_prec))));
        auto wgt_dram_width = std::max(dram->getBaseDataSize(), (uint32_t)pow(2, ceil(log2(wgt_prec))));

        *global_cycle = 0;
        dram->configure_layer(act_dram_width, wgt_dram_width);
        gbuffer->configure_layer();
        abuffer->configure_layer();
        pbuffer->configure_layer();
        wbuffer->configure_layer();
        obuffer->configure_layer();
        arch->configure_layer(act_prec, wgt_prec, ACT_BLKS, WGT_BLKS, -1, arch->diffy() || act->isSigned(),
                arch->diffy() || wgt->isSigned(), _linear, EF_COLUMNS);
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
    const std::vector<AddressRange> &Control<T>::getWriteAddresses() const {
        return on_chip_graph.front()->write_addresses;
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
    bool Control<T>::getIfLayerActOnChip() const {
        return on_chip_graph.front()->layer_act_on_chip;
    }

    template <typename T>
    bool Control<T>::still_off_chip_data() {
        on_chip_graph.erase(on_chip_graph.begin());
        return !on_chip_graph.empty();
    }

    template <typename T>
    bool Control<T>::check_if_write_output(const std::shared_ptr<TilesData<T>> &tiles_data) {
        for (const auto &tile_data : tiles_data->data)
            if (tile_data.write)
                return true;
        return false;
    }

    INITIALISE_DATA_TYPES(Control);

}
