
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

    uint32_t generateBoothEnconding(uint16_t n) {
        return 0;
    }

    std::vector<uint32_t> generateBoothTable() {
        std::vector<uint32_t> booth_table;
        for(long n = 0; n < 65536; n++)
            booth_table.push_back(generateBoothEnconding((uint16_t)n));
        return booth_table;
    }

    template <typename T>
    uint32_t Simulator<T>::booth_encoding(uint16_t value) {
        const static std::vector<uint32_t> booth_table = generateBoothTable();
        return booth_table[value];
    }

    INITIALISE_DATA_TYPES(Simulator);

}
