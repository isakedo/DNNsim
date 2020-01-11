#ifndef DNNSIM_UTILS_H
#define DNNSIM_UTILS_H

#include <base/Layer.h>
#include <base/Network.h>
#include <interface/NetReader.h>
#include <sys/common.h>

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

    template <typename T>
    class TileData {
    public:
        BufferSet<T> act_row;
        BufferRow<T> wgt_row;
        std::vector<WindowCoord> windows;
        std::vector<int> filters;
        int time = 0;
        int lanes = 0;
        bool valid = false;
    };

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

    /** Return value into sign-magnitude representation
     * @param two_comp  Signed value in two complement
     * @param mask      Mask with one bit for the bit position
     * @return          Value in sign-magnitude
     */
    uint16_t sign_magnitude(short two_comp, uint16_t mask);

}

#endif //DNNSIM_UTILS_H
