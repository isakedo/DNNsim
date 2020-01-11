#ifndef DNNSIM_STATISTICS_H
#define DNNSIM_STATISTICS_H

#include <sys/common.h>

namespace sys {

    /**
     * Return the average of a 1D vector
     * @tparam T Data type of the stat
     * @param vector_stat 1D Vector with the stats
     * @return Average of the vector
     */
    template <typename T>
    T get_average(const std::vector<T> &vector_stat)
    {
        return accumulate(vector_stat.begin(), vector_stat.end(), 0.0) / vector_stat.size();
    }

    /**
     * Return the average of a 2D vector
     * @tparam T Data type of the stat
     * @param vector_stat 2D Vector with the stats
     * @param skip_first Do not average first value
     * @return Average of the vector
     */
    template <typename T>
    T get_average(const std::vector<std::vector<T>> &vector_stat, bool skip_first = false)
    {
        std::vector<T> averages = std::vector<T>(vector_stat.size() - skip_first,0);
        for(int i = skip_first; i < vector_stat.size(); i++) {
            averages[i - skip_first] = get_average(vector_stat[i]);
        }
        return get_average(averages);
    }

    /**
     * Return the total of a 1D vector
     * @tparam T Data type of the stat
     * @param vector_stat 1D Vector with the stats
     * @return Total of the vector
     */
    template <typename T>
    T get_total(const std::vector<T> &vector_stat)
    {
        return accumulate(vector_stat.begin(), vector_stat.end(), 0.0);
    }

    /**
     * Return the total of a 2D vector
     * @tparam T Data type of the stat
     * @param vector_stat 2D Vector with the stats
     * @return Total of the vector
     */
    template <typename T>
    T get_total(const std::vector<std::vector<T>> &vector_stat)
    {
        std::vector<T> totals = std::vector<T>(vector_stat.size(), 0);
        for(uint64_t i = 0; i < vector_stat.size(); i++) {
            totals[i] = get_total(vector_stat[i]);
        }
        return get_total(totals);
    }

    /**
     * Return the sum of the averages of a 2D vector
     * @tparam T Data type of the stat
     * @param vector_stat 2D Vector with the stats
     * @return Sum of the averages of the vector
     */
    template <typename T>
    T get_average_total(const std::vector<std::vector<T>> &vector_stat)
    {
        std::vector<T> averages = std::vector<T>(vector_stat.size(), 0);
        for(uint64_t i = 0; i < vector_stat.size(); i++) {
            averages[i] = get_average(vector_stat[i]);
        }
        return get_total(averages);
    }

    /**
     * Return the minimum of a 1D vector
     * @tparam T Data type of the stat
     * @param vector_stat 1D Vector with the stats
     * @return Min value in the vector
     */
    template <typename T>
    T get_min(const std::vector<T> &vector_stat)
    {
        return *min_element(vector_stat.begin(), vector_stat.end());
    }

    /**
     * Return the minimum of a 2D vector
     * @tparam T Data type of the stat
     * @param vector_stat 2D Vector with the stats
     * @return Min value in the vector
     */
    template <typename T>
    T get_min(const std::vector<std::vector<T>> &vector_stat)
    {
        std::vector<T> mins = std::vector<T>(vector_stat.size(), 0);
        for(uint64_t i = 0; i < vector_stat.size(); i++) {
            mins[i] = get_min(vector_stat[i]);
        }
        return get_min(mins);
    }

    /**
     * Return the maximum of a 1D vector
     * @tparam T Data type of the stat
     * @param vector_stat 1D Vector with the stats
     * @return Max value in the vector
     */
    template <typename T>
    T get_max(const std::vector<T> &vector_stat)
    {
        return *max_element(vector_stat.begin(), vector_stat.end());
    }

    /**
     * Return the maximum of a 2D vector
     * @tparam T Data type of the stat
     * @param vector_stat 2D Vector with the stats
     * @return Max value in the vector
     */
    template <typename T>
    T get_max(const std::vector<std::vector<T>> &vector_stat)
    {
        std::vector<T> maxs = std::vector<T>(vector_stat.size(), 0);
        for(uint64_t i = 0; i < vector_stat.size(); i++) {
            maxs[i] = get_max(vector_stat[i]);
        }
        return get_max(maxs);
    }

    /**
    * Type of statistics
    */
    enum stat_type {
        Scalar,
        Distribution
    };

    /**
     * Measure for the images
     * Average: Average for images and layers
     * AverageTotal: Average for images and total across layers
     * Total: Total for images and layers
     * Max: Max across images and layers
     */
    enum Measure {
        No_Measure,
        Average,
        AverageTotal,
        Total,
        Max,
        Special
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
         * Special value for some stats
         */
        double special_value;

        /**
         * Special per epoch/layer value for some stats
         */
        std::vector<double> special_value_vector;

        /**
         * Skip first value when doing average
         */
        bool skip_first;

        /**
         * Constructor
         */
        stat_base_t();

        /**
         * Constructor
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_base_t(Measure _measure, bool _skip_first);

        /**
         * Destructor
         */
        virtual ~stat_base_t() = default;

        /**
         * Return the type of the stat
         */
        virtual stat_type getType() = 0;

        /**
         * Return the values of the stat in csv string format for specific image and layer
         * @param image image of the value
         * @param layer Layer of the value
         */
        virtual std::string to_string(uint64_t layer, uint64_t image) = 0;

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
     * String bits data type for the statistics.
     */
    class stat_string_t : public stat_base_t
    {

    public:

        /**
         * String values.
         */
        std::vector<std::vector<std::string>> value;

        /**
         * Constructor
         */
        stat_string_t() = default;

        /**
         * Constructor
         * @param _layers Number of layers
         * @param _images Number of images
         * @param _value Initial value
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_string_t(uint64_t _layers, uint64_t _images, const std::string &_value, Measure _measure,
                bool _skip_first);

        /**
         * Return scalar as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param image Index for the image
         */
        std::string to_string(uint64_t layer, uint64_t image) override;

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
         * @param _images Number of images
         * @param _value Initial value
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_uint_t(uint64_t _layers, uint64_t _images, uint64_t _value, Measure _measure, bool _skip_first);

        /**
         * Return scalar as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param image Index for the image
         */
        std::string to_string(uint64_t layer, uint64_t image) override;

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
         * @param _images Number of images
         * @param _value Initial value
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_double_t(uint64_t _layers, uint64_t _images, double _value, Measure _measure, bool _skip_first);

        /**
         * Return scalar as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param image Index for the image
         */
        std::string to_string(uint64_t layer, uint64_t image) override;

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
         * @param _images Number of images
         * @param _min_range Minimum value for the distribution range
         * @param _max_range Maximum value for the distribution range
         * @param _value Initial value
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_uint_dist_t(uint64_t _layers, uint64_t _images, uint64_t _min_range, uint64_t _max_range, uint64_t _value,
                Measure _measure, bool _skip_first);

        /**
         * Return distribution as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param image Index for the image
         */
        std::string to_string(uint64_t layer, uint64_t image) override;

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
         * @param _images Number of images
         * @param _min_range Minimum value for the distribution range
         * @param _max_range Maximum value for the distribution range
         * @param _value Initial value
         * @param _measure Measure for the statistics
         * @param _skip_first Skip first value when doing average
         */
        stat_double_dist_t(uint64_t _layers, uint64_t _images, uint64_t _min_range, uint64_t _max_range, double _value,
                Measure _measure, bool _skip_first);

        /**
         * Return distribution as type
         */
        stat_type getType() override;

        /**
         * Return the values of the stat in csv string format
         * @param layer Index for the layer
         * @param image Index for the image
         */
        std::string to_string(uint64_t layer, uint64_t image) override;

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
         * Number of images for the stats.
         */
        uint64_t images;

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
         * @param _images Number of images
         * @param _filename Name of the file
         */
        Stats(uint64_t _layers, uint64_t _images, const std::string &_filename);

        /**
         * Destructor
         */
        ~Stats() = default;

        /**
         * Register one string stat in the database.
         * @param name Name of the variable
         * @param measure Measure for the statistics
         * @param skip_first Skip first value when doing average
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_string_t> register_string_t(const std::string &name, Measure measure,
                bool skip_first = false);

        /**
         * Register one unsigned integer 64 bits stat in the database.
         * @param name Name of the variable
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @param skip_first Skip first value when doing average
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_uint_t> register_uint_t(const std::string &name, uint64_t init_value,  Measure measure,
                bool skip_first = false);

        /**
         * Register one double precision floating point stat in the database.
         * @param name Name of the variable
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @param skip_first Skip first value when doing average
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_double_t> register_double_t(const std::string &name, double init_value, Measure measure,
                bool skip_first = false);

        /**
         * Register one unsigned integer 64 bits distribution stat in the database.
         * @param name Name of the variable
         * @param min_range Minimum value for the distribution range
         * @param max_range Maximum value for the distribution range
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @param skip_first Skip first value when doing average
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_uint_dist_t> register_uint_dist_t(const std::string &name, int64_t min_range,
                int64_t max_range, uint64_t init_value, Measure measure, bool skip_first = false);

        /**
         * Register one double precision floating point distribution stat in the database.
         * @param name Name of the variable
         * @param min_range Minimum value for the distribution range
         * @param max_range Maximum value for the distribution range
         * @param init_value Initial value for the variable
         * @param measure Measure for the statistics
         * @param skip_first Skip first value when doing average
         * @return Reference to the registered stat
         */
        std::shared_ptr<stat_double_dist_t> register_double_dist_t(const std::string &name, int64_t min_range,
                int64_t max_range, double init_value, Measure measure, bool skip_first = false);

        /**
         * Return all stats per image in a csv file
         * @param network_name Name of the network
         * @param layers_name Name of the layers
         * @param header Header for the results
         * @param QUIET Avoid std::out messages
         */
        void dump_csv(const std::string &network_name, const std::vector<std::string> &layers_name,
                const std::string &header, bool QUIET);

    };

} //namespace sim

#endif //DNNSIM_STATISTICS_H
