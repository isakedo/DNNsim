#ifndef DNNSIM_BATCH_H
#define DNNSIM_BATCH_H

#include <sys/common.h>
#include <batch.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

namespace sys {

    class Batch {

    private:

        /* Struct for the Transform instructions */
        struct Transform {
            std::string inputType = ""; // Caffe/Protobuf/Gzip
            std::string inputDataType = "";
            std::string outputType = ""; // Protobuf/Gzip
            std::string outputDataType = ""; // Protobuf/Gzip
            std::string network = "";
            bool activate_bias_out_act = false;
        };

        /* Struct for the Simulate instructions */
        struct Simulate {

            struct Experiment {
                std::string architecture = "";
                std::string task = "";
                int n_columns = 0;
                int n_rows = 0;
                std::string precision_granularity = "";
                int bits_first_stage = 0;
                int lookahead_h = 0;
                int lookaside_d = 0;
                char search_shape = 'X';
                bool read_schedule_from_proto = false;
            };

            std::string inputType = ""; // Protobuf/Gzip
            std::string inputDataType = ""; // Float32/Fixed16
            std::string network = "";
            bool activate_bias_out_act = false;
            std::vector<Experiment> experiments;
        };

        /* Path to the batch file */
        std::string path;

        /* Transformations */
        std::vector<Transform> transformations;

        /* Simulations */
        std::vector<Simulate> simulations;

        /* Return the transformation parsed from the prototxt file
         * @param transform_proto   prototxt transformation
         */
        Transform read_transformation(const protobuf::Batch_Transform &transform_proto);

        /* Return the simulation parsed from the prototxt file
         * @param simulate_proto   prototxt simulation
         */
        Simulate read_simulation(const protobuf::Batch_Simulate &simulate_proto);

    public:

        /* Constructor
         * @param _path     Path to the batch file
         */
        explicit Batch(const std::string &_path){ this->path = _path; }

        /* Parse the batch file into memory */
        void read_batch();

        /* Getters */
        const std::vector<Transform> &getTransformations() const;
        const std::vector<Simulate> &getSimulations() const;

    };

}


#endif //DNNSIM_BATCH_H
