#ifndef DNNSIM_BATCH_H
#define DNNSIM_BATCH_H

#include <sys/common.h>
#include <batch.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

namespace sys {

    class Batch {

    public:

        /* Struct for the Simulate instructions */
        struct Simulate {

            struct Experiment {
                std::string architecture = "";
                std::string task = "";
                uint32_t n_lanes = 0;
                uint32_t n_columns = 0;
                uint32_t n_rows = 0;
                uint32_t bits_pe = 0;
                uint32_t column_registers = 0;
                uint32_t precision_granularity = 0;
                bool leading_bit = false;
                bool minor_bit = false;
                uint32_t bits_first_stage = 0;
                uint32_t lookahead_h = 0;
                uint32_t lookaside_d = 0;
                char search_shape = 'X';
                bool read_schedule = false;
                uint32_t Wt = 0;
                uint32_t Ht = 0;
                uint32_t I = 0;
                uint32_t F = 0;
                uint32_t out_acc_size = 0;
                uint32_t banks = 0;
                uint32_t pe_serial_bits = 0;
                uint32_t M = 0;
                uint32_t N = 0;
                uint32_t pmax = 0;
                uint32_t pmin = 0;
                bool diffy = false;
                bool dynamic_weights = false;
            };

            uint32_t batch = 0;
            uint32_t epochs = 0;
            std::string model = "";
            std::string data_type = "";
            std::string network = "";
            uint32_t network_bits = 0;
            bool tensorflow_8b = false;
            bool training = false;
            bool only_forward = false;
            bool only_backward = false;
            uint32_t decoder_states = 0;
            std::vector<Experiment> experiments;
        };

    private:

        /* Path to the batch file */
        std::string path;

        /* Simulations */
        std::vector<Simulate> simulations;

        /* Return the training simulation parsed from the prototxt file
         * @param simulate_proto   prototxt simulation
         */
        Simulate read_training_simulation(const protobuf::Batch_Simulate &simulate_proto);

        /* Return the inference simulation parsed from the prototxt file
         * @param simulate_proto   prototxt simulation
         */
        Simulate read_inference_simulation(const protobuf::Batch_Simulate &simulate_proto);

    public:

        /* Constructor
         * @param _path     Path to the batch file
         */
        explicit Batch(const std::string &_path){ this->path = _path; }

        /* Parse the batch file into memory */
        void read_batch();

        /* Getters */
        const std::vector<Simulate> &getSimulations() const;

    };

}


#endif //DNNSIM_BATCH_H
