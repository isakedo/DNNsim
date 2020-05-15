#ifndef DNNSIM_BATCH_H
#define DNNSIM_BATCH_H

#include <sys/common.h>
#include <batch.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace sys {

    /**
     * Batch configuration
     */
    class Batch {

    public:

        /** Struct for the Simulate instructions */
        struct Simulate {

            /** Struct for the Experiments configurations */
            struct Experiment {

                /** Architecture name */
                std::string architecture = "";

                /** Task name */
                std::string task = "";

                /** Dataflow name */
                std::string dataflow = "";

                /** Number of lanes */
                uint32_t lanes = 0;

                /** Number of columns */
                uint32_t columns = 0;

                /** Number of rows */
                uint32_t rows = 0;

                /** Number of tiles */
                uint32_t tiles = 0;

                /** Bits per PE */
                uint32_t pe_width = 0;

                /** Column registers */
                uint32_t column_registers = 0;

                /** Booth-like encoding */
                bool booth = false;

                /** Group size */
                uint32_t group_size = 0;

                /** Minor bit */
                bool minor_bit = false;

                /** Bit first shift stage: Pragmatic PE */
                uint32_t bits_first_stage = 0;

                /** Lookahead H: Tactical */
                uint32_t lookahead_h = 0;

                /** Lookaside D: Tactical */
                uint32_t lookaside_d = 0;

                /** Search shape: Tactical */
                std::string search_shape = "X";

                /** X Dim PEs: SCNN */
                uint32_t Wt = 0;

                /** Y Dim PEs: SCNN */
                uint32_t Ht = 0;

                /** X Dim multiplier pe PER: SCNN */
                uint32_t I = 0;

                /** Y Dim multiplier pe PER: SCNN */
                uint32_t F = 0;

                /** Output accumulator size: SCNN */
                uint32_t out_acc_size = 0;

                /** Output accumulator banks: SCNN */
                uint32_t banks = 0;

                /** Serial bits per PE */
                uint32_t pe_serial_bits = 0;

                /** Diffy simulation */
                bool diffy = false;

                /** BitTactical simulation */
                bool tactical = false;

                /** Dynamic width precision: Loom */
                bool dynamic_weights = false;

                /** CPU clock frequency */
                uint64_t cpu_clock_freq = 0;

                /** Dram size */
                uint64_t dram_size = 0;

                /** DRAM start activations address */
                uint64_t dram_start_act_address = 0;

                /** DRAM start weights address */
                uint64_t dram_start_wgt_address = 0;

                /** Global buffer activation memory size */
                uint32_t gbuffer_act_size = 0;

                /** Global buffer weight memory size */
                uint32_t gbuffer_wgt_size = 0;

                /** Global buffer act banks */
                uint32_t gbuffer_act_banks = 0;

                /** Global buffer wgt banks */
                uint32_t gbuffer_wgt_banks = 0;

                /** Global buffer out banks */
                uint32_t gbuffer_out_banks = 0;

                /** Global buffer bank width */
                uint32_t gbuffer_bank_width = 0;

                /** Global buffer read delay */
                uint32_t gbuffer_read_delay = 0;

                /** Global buffer write delay */
                uint32_t gbuffer_write_delay = 0;

                /** Activation buffer number of memory rows */
                uint32_t abuffer_rows = 0;

                /** Activation buffer read delay */
                uint32_t abuffer_read_delay = 0;

                /** Weight buffer number of memory rows */
                uint32_t wbuffer_rows = 0;

                /** Weight buffer read delay */
                uint32_t wbuffer_read_delay = 0;

                /** Output buffer number of memory rows */
                uint32_t obuffer_rows = 0;

                /** Output buffer write delay */
                uint32_t obuffer_write_delay = 0;

                /** Composer column inputs per tile */
                uint32_t composer_inputs = 0;

                /** Composer column delay */
                uint32_t composer_delay = 0;

                /** Post-Processing Unit inputs */
                uint32_t ppu_inputs = 0;

                /** Post-Processing Unit delay */
                uint32_t ppu_delay = 0;

            };

            /** Batch number of the traces */
            uint32_t batch = 0;

            /** Model input type */
            std::string model = "";

            /** Data input type */
            std::string data_type = "";

            /** Network name */
            std::string network = "";

            /** Network data width */
            uint32_t data_width = 0;

            /** Array of experiments */
            std::vector<Experiment> experiments;
        };

    private:

        /** Path to the batch file */
        std::string path;

        /** Simulations */
        std::vector<Simulate> simulations;

        /** Return the inference simulation parsed from the prototxt file
         * @param simulate_proto   prototxt simulation
         * @return Simulate configuration
         */
        Simulate read_inference_simulation(const protobuf::Batch_Simulate &simulate_proto);

    public:

        /** Constructor
         * @param _path     Path to the batch file
         */
        explicit Batch(const std::string &_path){ this->path = _path; }

        /** Parse the batch file into memory */
        void read_batch();

        /**
         * Get all simulate configurations
         * @return Array of simulate configurations
         */
        const std::vector<Simulate> &getSimulations() const;

    };

}


#endif //DNNSIM_BATCH_H
