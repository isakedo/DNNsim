#ifndef DNNSIM_UTILS_H
#define DNNSIM_UTILS_H

#include <base/Network.h>

namespace core {

    typedef std::vector<std::vector<std::vector<double>>> OutputTensor;

    typedef std::tuple<uint16_t, uint16_t> ValueIndex;

    template <typename T>
    using ValueTuple = std::tuple<T, uint16_t, uint16_t>;

    template <typename T>
    using BufferRow = std::vector<ValueTuple<T>>;

    template <typename T>
    using BufferSet = std::vector<std::vector<ValueTuple<T>>>;

    template <typename T>
    using Buffer = std::vector<std::vector<std::vector<ValueTuple<T>>>>;

    typedef std::tuple<int, int> WindowCoord;

    typedef std::tuple<uint64_t, uint64_t> AddressRange;

    typedef std::vector<std::vector<std::vector<uint64_t>>> AddressMap;

    typedef std::vector<std::vector<int>> ActBankMap;

    typedef std::vector<std::vector<std::vector<uint64_t>>> AddressBuffer;

    typedef std::vector<std::vector<uint64_t>> AddressBufferSet;

    typedef std::vector<uint64_t> AddressBufferRow;

    typedef std::vector<std::vector<std::vector<int>>> BankBuffer;

    typedef std::vector<std::vector<int>> BankBufferSet;

    typedef std::vector<int> BankBufferRow;

    /**
     * Data to process per tile
     * @tparam T Data type values
     */
    template <typename T>
    class TileData {
    public:

        /** 2D Input activations (2D because of Tactical) */
        BufferSet<T> act_row;

        /** 1D Input weights */
        BufferRow<T> wgt_row;

        /** Window indices to process */
        std::vector<WindowCoord> windows;

        /** Filter indices to process */
        std::vector<int> filters;

        /** 2D Input activation mapped addresses */
        AddressBufferSet act_addresses;

        /** 1D Weights mapped addresses */
        AddressBufferRow wgt_addresses;

        /** 1D Partial sum mapped addresses */
        AddressBufferRow psum_addresses;

        /** 1D Output activations mapped addresses */
        AddressBufferRow out_addresses;

        /** 2D Input activation mapped on-chip banks */
        BankBufferSet act_banks;

        /** 1D Partial sum mapped on-chip banks */
        BankBufferRow psum_banks;

        /** 1D Weight mapped on-chip banks */
        BankBufferRow wgt_banks;

        /** 1D Output activation mapped on-chip banks */
        BankBufferRow out_banks;

        /** Current time in the 2D input buffer (for Tactical) */
        int time = 0;

        /** Total number of lines */
        int lanes = 0;

        /** Valida data flag */
        bool valid = false;
    };

    /**
     * Data to process
     * @tparam T Data type values
     */
    template <typename T>
    class TilesData {
    public:

        /** Data to process per tile */
        std::vector<TileData<T>> data;

        /** Read activations flag */
        bool read_act = false;

        /** Read partial sum flag */
        bool read_psum = false;

        /** Read weights flag */
        bool read_wgt = false;

        /**
         * Constructor
         * @param _tiles Total number of tiles
         */
        explicit TilesData(uint64_t _tiles) {
            data = std::vector<TileData<T>>(_tiles, TileData<T>());
        }
    };

    /**
     * Transform the memory size to text
     * @param mem Memory size integer
     * @return Memory size text
     */
    std::string to_mem_string(uint64_t mem);

    /** Return the optimal encoding for the given value
     * @param value     Value to encode WITHOUT the sign
     * @return          Value with the optimal encoding
     */
    uint16_t booth_encoding(uint16_t value);

    /** Return the minimum and maximum index position for a given value
     * @param value     Value to get the indexes
     * @return          Minimum and maximum indexes
     */
    std::tuple<uint8_t,uint8_t> minMax(uint16_t value);

    /** Return the number of effectual bits for a given value
     * @param value     Value to get the effectual bits
     * @return          Number of effectual bits
     */
    uint8_t effectualBits(uint16_t value);

}

#endif //DNNSIM_UTILS_H
