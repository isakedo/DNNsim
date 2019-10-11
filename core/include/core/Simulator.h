#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include <sys/common.h>
#include <sys/Stats.h>
#include <sys/Batch.h>
#include <base/Array.h>
#include <base/Layer.h>
#include <base/Network.h>
#include <interface/NetReader.h>
#include <core/Memory.h>

#ifdef OPENMP
#include <omp.h>
#endif

typedef std::vector<std::vector<std::vector<std::vector<uint64_t>>>> address_map;

namespace core {

    /**
     * Base class simulator
     * @tparam T Data type of the simulation
     */
    template <typename T>
    class Simulator {

    private:

        typedef union {
            float f;
            struct {
                unsigned int truncated_mantissa : 16;
                unsigned int mantissa : 7;
                unsigned int exponent : 8;
                unsigned int sign : 1;
            } field;
        } bfloat16;

    protected:

        /** Memory abstraction of the simulator */
        Memory memory;

        /** Number of parallel cores */
        const int N_THREADS;

        /** Enable fast mode: only one image */
        const bool FAST_MODE = false;

        /** Avoid std::out messages */
        const bool QUIET = false;

        /** Check the correctness of the simulations */
        const bool CHECK = false;

        /** Read training traces for a given epoch
         * @param network_name      Name of the network
         * @param batch             Batch of the traces
         * @param epoch             Epoch of the traces
         * @param decoder_states    Number of states of the decoder
         * @param traces_mode       Fordward/Backward traces
         */
        base::Network<T> read_training(const std::string &network_name, uint32_t batch, uint32_t epoch,
                uint32_t decoder_states, uint32_t traces_mode);

        /** Iterate set of windows in groups
         * @param out_x         Output activations X size
         * @param out_y         Output activations Y size
         * @param list_x        X position for the set of input windows (Overwritten)
         * @param list_y        Y position for the set of input windows (Overwritten)
         * @param x_counter     X input window counter (Overwritten)
         * @param y_counter     Y input window counter (Overwritten)
         * @param max_windows   Maximum number of windows (Number of columns in the accelerator)
         * @return              Return false when all input windows are read
         */
        bool iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
                int &x_counter, int &y_counter, int max_windows = 16);

        /** Split sign, exponent, and mantissa in bfloat16 format from a float
         * @param number    Floating point 32 number
         * return           Tuple containing sign, exponent, and mantissa (truncated)
         */
        std::tuple<uint8_t,uint8_t,uint8_t> split_bfloat16(float number);

        /** Return floating-point single precision number in bfloat 16
         * @param number    Floating point 32 number
         * return           BFloat 16 number
         */
        float cast_bfloat16(float number);

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

        /** Return true if all the queues of activation bits are empty
         * @param offsets   Collection of activations with their explicit one positions in a queue
         * @return          True if empty
         */
        bool check_act_bits(const std::vector<std::queue<uint8_t>> &offsets);

        /** Return value into sign-magnitude representation
         * @param two_comp  Signed value in two complement
         * @param mask      Mask with one bit for the bit position
         * @return          Value in sign-magnitude
         */
        uint16_t sign_magnitude(short two_comp, uint16_t mask);

    public:

        /** Constructor
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        Simulator(uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : N_THREADS(_N_THREADS),
                FAST_MODE(_FAST_MODE), QUIET(_QUIET), CHECK(_CHECK), memory(Memory()) {}


       /** Returns network information
        * @param network   Network we want to check
        */
        void information(const base::Network<T> &network);

        /** Calculate the sparsity in the network
         * @param network   Network we want to check
         */
        void sparsity(const base::Network<T> &network);

        /** Calculate the bit-sparsity in the network. Assumes two-complement
         * @param network   Network we want to check
         */
        void bit_sparsity(const base::Network<T> &network);

        /** Calculate the training sparsity in the network
         * @param simulate  Simulate configuration
		 * @param epochs    Number of epochs
         */
        void training_sparsity(const sys::Batch::Simulate &simulate, int epochs);

        /** Calculate the training bit sparsity in the network
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         * @param mantissa  Mantissa bit sparsity instead of exponent
         */
        void training_bit_sparsity(const sys::Batch::Simulate &simulate, int epochs, bool mantissa);

        /** Calculate the training exponent distribution in the network
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         * @param mantissa  Mantissa distribution instead of exponent
         */
        void training_distribution(const sys::Batch::Simulate &simulate, int epochs, bool mantissa);

    };

}

#endif //DNNSIM_SIMULATOR_H
