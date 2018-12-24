
#include <core/Simulator.h>

namespace core {

    template <typename T>
    cnpy::Array<T> Simulator<T>::adjustPadding(const cnpy::Array<T> &array, int padding) {
        cnpy::Array<T> padded_array;
        std::vector<T> padded_data;
        const auto &shape = array.getShape();

        for(int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = -padding; k < (int)shape[2] + padding; k++) {
                    if(k >= 0 && k < shape[2]) {
                        for (int p = 0; p < padding; p++) padded_data.push_back(0);
                        for (int l = 0; l < shape[3]; l++) padded_data.push_back(array.get(i, j, k, l));
                        for (int p = 0; p < padding; p++) padded_data.push_back(0);
                    } else {
                        for (int l = -padding; l < (int)shape[3] + padding; l++) padded_data.push_back(0);
                    }
                }
            }
        }

        std::vector<size_t > padded_shape;
        padded_shape.push_back(shape[0]);
        padded_shape.push_back(shape[1]);
        padded_shape.push_back(shape[2] + 2*padding);
        padded_shape.push_back(shape[3] + 2*padding);
        padded_array.set_values(padded_data,padded_shape);
        return padded_array;
    }

    template <typename T>
    bool Simulator<T>::iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
            int max_windows) {
        static int x = 0;
        static int y = 0;
        list_x.clear();
        list_y.clear();
        int current_windows = 0;
        while(x < out_x) {
            while(y < out_y) {
                list_x.push_back(x);
                list_y.push_back(y);
                current_windows++;
                y++;
                if(current_windows >= max_windows)
                    return true;
            }
            y = 0;
            x++;
        }
        if(current_windows > 0)
            return true;

        x = 0;
        return false;
    }

    /* Only encode the values when get less number of bits */
    uint16_t generateBoothEnconding(uint16_t n) {
        uint32_t padded_n = n << 2;
        std::string bitstream = std::bitset<16 + 2>(padded_n).to_string();
        uint16_t booth_encoding = 0;
        bool booth = false;
        for(int i = 0; i < 16; i++) {
            std::string w = bitstream.substr(0,3);
            booth_encoding <<= 1;
            if(w == "000" || w == "001") {
                assert(!booth);
            } else if(w == "010") {
                if (booth) booth_encoding |= 0x1;
            } else if(w == "011") {
                if (booth) booth_encoding |= 0x1;
            } else if(w == "100") {
                if (!booth) booth_encoding |= 0x1;
                else { booth_encoding |= 0x1; booth = false;}
            } else if(w == "101") {
                if (!booth) booth_encoding |= 0x1;
            } else if(w == "110") {
                if (!booth) booth_encoding |= 0x1;
            } else if(w == "111") {
                if (!booth) { booth_encoding |= 0x2;  booth = true; }
            }
            bitstream = bitstream.substr(1);
        }
        return booth_encoding;
    }

    std::vector<uint16_t> generateBoothTable() {
        std::vector<uint16_t> booth_table;
        for(long n = 0; n < 32768; n++)
            booth_table.push_back(generateBoothEnconding((uint16_t)n));
        return booth_table;
    }

    template <typename T>
    uint16_t Simulator<T>::booth_encoding(uint16_t value) {
        const static std::vector<uint16_t> booth_table = generateBoothTable();
        return booth_table[value];
    }

    INITIALISE_DATA_TYPES(Simulator);

}
