#ifndef DNNSIM_STATISTICS_H
#define DNNSIM_STATISTICS_H

#include <sys/common.h>

namespace sys {

    /**
    * Type of statistics
    */
    enum stat_type {
        Scalar,
        Distribution
    };

    /**
     * Measure for the batches
     * Average: Average for batches and layers
     * AverageTotal: Average for batches and total across layers
     * Total: Total for batches and layers
     * Max: Max across batches and layers
     */
    enum Measure {
        No_Measure,
        Average,
        AverageTotal,
        Total,
        Max
    };

    /**
     * Abstract data type for the statistics.
     */
    class stat_base_t
    {

    public:

        /**
         * Measure for the statistics
         */
        Measure measure;

        /**
         * Constructor
         */
        stat_base_t();

        /**
         * Constructor
         * @param _measure Measure for the statistics
         */
        explicit stat_base_t(Measure _measure);

        /**
         * Destructor
         */
        virtual ~stat_base_t() = default;

        /**
         * Return the type of the stat
         */
        virtual stat_type getType() = 0;

        /**
         * Return the values of the stat in csv string format for specific batch and layer
         * @param batch Batch of the value
         * @param layer Layer of the value
         */
        virtual std::string to_string(uint64_t layer, uint64_t batch) = 0;

        /**
         * Return the values of the stat in csv string format for specific layer
         * @param layer Layer of the value
         */
        virtual std::string layer_to_string(uint64_t layer) = 0;

        /**
         * Return the values of the stat in csv string format for the whole network
         */
        virtual std::string network_to_string() = 0;

        /**
         * Return the header range for a distribution
         */
        virtual std::string dist_to_string() = 0;

    };

    /**
     * Unsigned integer 64 bits data type for the statistics.
     */
    class stat_uint_t : public stat_base_t
    {

    public:

        /**
         * Unsigned 64 bits values.
         */
        std::vector<std::vector<uint64_t>> value;

        /**
         * Constructor
         */
        stat_uint_t() = default;

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _batches Number of batches
         * @param _value Initial value
         * @param _measure Measure for the statistics
         */
        stat_uint_t(uint64_t _layers, uint64_t _batches, uint64_t _value, Measure _measure);

        /**
         * Return scalar as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param batch Index for the batch
         */
        std::string to_string(uint64_t layer, uint64_t batch) override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         */
        std::string layer_to_string(uint64_t layer) override;

        /**
         * Return the values of the stat in csv string format for the whole network
         */
        std::string network_to_string() override;

        /**
         * Return the header range for a distribution
         */
        std::string dist_to_string() override;

    };

    /**
     * Double precision floating data type for the statistics.
     */
    class stat_double_t : public stat_base_t
    {

    public:

        /**
         * Double precision floating point values.
         */
        std::vector<std::vector<double>> value;

        /**
         * Constructor
         */
        stat_double_t() = default;

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _batches Number of batches
         * @param _value Initial value
         * @param _measure Measure for the statistics
         */
        stat_double_t(uint64_t _layers, uint64_t _batches, double _value, Measure _measure);

        /**
         * Return scalar as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param batch Index for the batch
         */
        std::string to_string(uint64_t layer, uint64_t batch) override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         */
        std::string layer_to_string(uint64_t layer) override;

        /**
         * Return the values of the stat in csv string format for the whole network
         */
        std::string network_to_string() override;

        /**
         * Return the header range for a distribution
         */
        std::string dist_to_string() override;

    };

    /**
     * Unsigned integer 64 bits distribution data type for the statistics.
     */
    class stat_uint_dist_t : public stat_base_t
    {

    public:

        /**
         * Array of unsigned 64 bits values.
         */
        std::vector<std::vector<std::vector<uint64_t>>> value;

        /**
         * Minimum value for the distribution range
         */
        int64_t min_range;

        /**
         * Maximum value for the distribution range
         */
        int64_t max_range;

        /**
         * Constructor
         */
        stat_uint_dist_t();

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _batches Number of batches
         * @param _min_range Minimum value for the distribution range
         * @param _max_range Maximum value for the distribution range
         * @param _value Initial value
         * @param _measure Measure for the statistics
         */
        stat_uint_dist_t(uint64_t _layers, uint64_t _batches, uint64_t _min_range, uint64_t _max_range, uint64_t _value,
                Measure _measure);

        /**
         * Return distribution as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param batch Index for the batch
         */
        std::string to_string(uint64_t layer, uint64_t batch) override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         */
        std::string layer_to_string(uint64_t layer) override;

        /**
         * Return the values of the stat in csv string format for the whole network
         */
        std::string network_to_string() override;

        /**
         * Return the header range for a distribution
         */
        std::string dist_to_string() override;

    };

    /**
     * Double precision floating point distribution data type for the statistics.
     */
    class stat_double_dist_t : public stat_base_t
    {

    public:

        /**
         * Array of double precision floating point values.
         */
        std::vector<std::vector<std::vector<double>>> value;

        /**
         * Minimum value for the distribution range
         */
        int64_t min_range;

        /**
         * Maximum value for the distribution range
         */
        int64_t max_range;

        /**
         * Constructor
         */
        stat_double_dist_t();

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _batches Number of batches
         * @param _min_range Minimum value for the distribution range
         * @param _max_range Maximum value for the distribution range
         * @param _value Initial value
         * @param _measure Measure for the statistics
         */
        stat_double_dist_t(uint64_t _layers, uint64_t _batches, uint64_t _min_range, uint64_t _max_range, double _value,
                Measure _measure);

        /**
         * Return distribution as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param batch Index for the batch
         */
        std::string to_string(uint64_t layer, uint64_t batch) override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         */
        std::string layer_to_string(uint64_t layer) override;

        /**
         * Return the values of the stat in csv string format for the whole network
         */
        std::string network_to_string() override;

        /**
         * Return the header range for a distribution
         */
        std::string dist_to_string() override;

    };

    /**
     * Class containing stats for the simulator
     */
    class Stats
    {

    private:

        /**
         * Struct for the stats definition
         */
        struct table_t
        {

            /**
             * Name of the stat
             */
            std::string name;

            /**
             * Value of the stat
             */
            std::shared_ptr<sys::stat_base_t> var;

            ~table_t() = default;

        };

        /**
         * Database for all the stats in the simulation
         */
        std::vector<table_t> database;

        /**
         * Number of layers for the stats.
         */
        uint64_t layers;

        /**
         * Number of batches for the stats.
         */
        uint64_t batches;

        /**
         * Name of the file
         */
        std::string filename;

        /**
         * Check if the path exists
         * @param path Path to check
         */
        static void check_path(const std::string &path);

    public:

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _batches Number of batches
         * @param _filename Name of the file
         */
        Stats(uint64_t _layers, uint64_t _batches, const std::string &_filename);

        /**
         * Destructor
         */
        ~Stats() = default;

        /**
         * Register one unsigned integer 64 bits stat in the database.
         * @param name Name of the variable
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_uint_t> register_uint_t(const std::string &name, uint64_t init_value,  Measure measure);

        /**
         * Register one double precision floating point stat in the database.
         * @param name Name of the variable
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_double_t> register_double_t(const std::string &name, double init_value, Measure measure);

        /**
         * Register one unsigned integer 64 bits distribution stat in the database.
         * @param name Name of the variable
         * @param min_range Minimum value for the distribution range
         * @param max_range Maximum value for the distribution range
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_uint_dist_t> register_uint_dist_t(const std::string &name, int64_t min_range,
                int64_t max_range, uint64_t init_value, Measure measure);

        /**
         * Register one double precision floating point distribution stat in the database.
         * @param name Name of the variable
         * @param min_range Minimum value for the distribution range
         * @param max_range Maximum value for the distribution range
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_double_dist_t> register_double_dist_t(const std::string &name, int64_t min_range,
                int64_t max_range, double init_value, Measure measure);

        /**
         * Return all stats per image in a csv file
         * @param network Name of the network
         * @param layers Name of the layers
         * @param QUIET Avoid std::out messages
         */
        void dump_csv(const std::string &network_name, const std::vector<std::string> &layers_name, bool QUIET);

    };

} //namespace sim

#endif //DNNSIM_STATISTICS_H
