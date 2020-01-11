
#include <sys/Stats.h>
#include <sys/stat.h>

namespace sys {

    //stat_base_t

    stat_base_t::stat_base_t() : measure(No_Measure), special_value(0.0), skip_first(false) {}

    stat_base_t::stat_base_t(Measure _measure, bool _skip_first) : measure(_measure), skip_first(_skip_first),
            special_value(0.0), special_value_vector() {}

    // string_t

    stat_string_t::stat_string_t(uint64_t _layers, uint64_t _images, const std::string &_value, Measure _measure,
            bool _skip_first) : stat_base_t(_measure,_skip_first)
    {
        value = std::vector<std::vector<std::string>>(_layers, std::vector<std::string>(_images, _value));
    }

    inline stat_type stat_string_t::getType()
    {
        return stat_type::Scalar;
    }

    inline std::string stat_string_t::to_string(uint64_t layer, uint64_t image)
    {
        return value[layer][image];
    }

    inline std::string stat_string_t::layer_to_string(uint64_t layer)
    {
        return value[layer][0];
    }

    inline std::string stat_string_t::network_to_string()
    {
        return "-";
    }

    inline std::string stat_string_t::dist_to_string()
    {
        throw std::runtime_error("Wrong stat type");
    }

    // stat_uint_t

    stat_uint_t::stat_uint_t(uint64_t _layers, uint64_t _images, uint64_t _value, Measure _measure, bool _skip_first)
            : stat_base_t(_measure,_skip_first)
    {
        value = std::vector<std::vector<uint64_t>>(_layers, std::vector<uint64_t>(_images, _value));
    }

    inline stat_type stat_uint_t::getType()
    {
        return stat_type::Scalar;
    }

    inline std::string stat_uint_t::to_string(uint64_t layer, uint64_t image)
    {
        return std::to_string(value[layer][image]);
    }

    inline std::string stat_uint_t::layer_to_string(uint64_t layer)
    {
        if (measure == Measure::Average || measure == Measure::AverageTotal || measure == Measure::Special) {
            return std::to_string(get_average(value[layer]));
        } else if (measure == Measure ::Total) {
            return std::to_string(get_total(value[layer]));
        } else if (measure == Measure ::Max) {
            return std::to_string(get_max(value[layer]));
        } else {
            throw std::runtime_error("Wrong measure formula");
        }
    }

    inline std::string stat_uint_t::network_to_string()
    {
        if (measure == Measure::Average) {
            return std::to_string(get_average(value,skip_first));
        } else if ( measure == Measure::AverageTotal) {
            return std::to_string(get_average_total(value));
        } else if (measure == Measure ::Total) {
            return std::to_string(get_total(value));
        } else if (measure == Measure ::Max) {
            return std::to_string(get_max(value));
        } else {
            throw std::runtime_error("Wrong measure formula");
        }
    }

    inline std::string stat_uint_t::dist_to_string()
    {
        throw std::runtime_error("Wrong stat type");
    }

    // stat_double_t

    stat_double_t::stat_double_t(uint64_t _layers, uint64_t _images, double _value, Measure _measure, bool _skip_first) :
            stat_base_t(_measure,_skip_first)
    {
        value = std::vector<std::vector<double>>(_layers, std::vector<double>(_images, _value));
    }

    inline stat_type stat_double_t::getType()
    {
        return stat_type::Scalar;
    }

    inline std::string stat_double_t::to_string(uint64_t layer, uint64_t image)
    {
        return std::to_string(value[layer][image]);
    }

    inline std::string stat_double_t::layer_to_string(uint64_t layer)
    {
        if (measure == Measure::Average || measure == Measure::AverageTotal || measure == Measure::Special) {
            return std::to_string(get_average(value[layer]));
        } else if (measure == Measure ::Total) {
            return std::to_string(get_total(value[layer]));
        } else if (measure == Measure ::Max) {
            return std::to_string(get_max(value[layer]));
        } else {
            throw std::runtime_error("Wrong measure formula");
        }
    }

    inline std::string stat_double_t::network_to_string()
    {
        if (measure == Measure::Average) {
            return std::to_string(get_average(value,skip_first));
        } else if ( measure == Measure::AverageTotal) {
            return std::to_string(get_average_total(value));
        } else if (measure == Measure ::Total) {
            return std::to_string(get_total(value));
        } else if (measure == Measure ::Max) {
            return std::to_string(get_max(value));
        } else {
            throw std::runtime_error("Wrong measure formula");
        }
    }

    inline std::string stat_double_t::dist_to_string()
    {
        throw std::runtime_error("Wrong stat type");
    }

    // stat_uint_dist_t

    stat_uint_dist_t::stat_uint_dist_t() : min_range(0), max_range(0) {}

    stat_uint_dist_t::stat_uint_dist_t(uint64_t _layers, uint64_t _images, uint64_t _min_range, uint64_t _max_range,
            uint64_t _value, Measure _measure, bool _skip_first) : min_range(_min_range), max_range(_max_range),
            stat_base_t(_measure,_skip_first)
    {
        auto _n_values = max_range - min_range + 1;
        value = std::vector<std::vector<std::vector<uint64_t>>>(_n_values, std::vector<std::vector<uint64_t>>
                (_layers, std::vector<uint64_t>(_images, _value)));
    }

    inline stat_type stat_uint_dist_t::getType()
    {
        return stat_type::Distribution;
    }

    inline std::string stat_uint_dist_t::to_string(uint64_t layer, uint64_t image) {
        std::string line;
        for (const auto &_value : value) {
            line += std::to_string(_value[layer][image]) + ',';
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_uint_dist_t::layer_to_string(uint64_t layer)
    {
        std::string line;
        for (const auto &_value : value) {
            if (measure == Measure::Average || measure == Measure::AverageTotal || measure == Measure::Special) {
                line += std::to_string(get_average(_value[layer])) + ',';
            } else if (measure == Measure::Total) {
                line += std::to_string(get_total(_value[layer])) + ',';
            } else if (measure == Measure::Max) {
                line += std::to_string(get_max(_value[layer])) + ',';
            } else {
                throw std::runtime_error("Wrong measure formula");
            }
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_uint_dist_t::network_to_string()
    {
        std::string line;
        for (const auto &_value : value) {
            if (measure == Measure::Average) {
                line += std::to_string(get_average(_value,skip_first)) + ',';
            } else if (measure == Measure::AverageTotal) {
                line += std::to_string(get_average_total(_value)) + ',';
            }else if (measure == Measure::Total) {
                line += std::to_string(get_total(_value)) + ',';
            } else if (measure == Measure::Max) {
                line += std::to_string(get_max(_value)) + ',';
            } else {
                throw std::runtime_error("Wrong measure formula");
            }
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_uint_dist_t::dist_to_string()
    {
        std::string line;
        for (int64_t r = min_range; r <= max_range; ++r) {
            line += std::to_string(r) + ',';
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    // stat_double_dist_t

    stat_double_dist_t::stat_double_dist_t() : min_range(0), max_range(0) {}

    stat_double_dist_t::stat_double_dist_t(uint64_t _layers, uint64_t _images, uint64_t _min_range, uint64_t _max_range,
            double _value, Measure _measure, bool _skip_first) : min_range(_min_range), max_range(_max_range),
            stat_base_t(_measure,_skip_first)
    {
        auto _n_values = max_range - min_range + 1;
        value = std::vector<std::vector<std::vector<double>>>(_n_values, std::vector<std::vector<double>>
                (_layers, std::vector<double>(_images, _value)));
    }

    inline stat_type stat_double_dist_t::getType()
    {
        return stat_type::Distribution;
    }

    inline std::string stat_double_dist_t::to_string(uint64_t layer, uint64_t image) {
        std::string line;
        for (const auto &_value : value) {
            line += std::to_string(_value[layer][image]) + ',';
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_double_dist_t::layer_to_string(uint64_t layer)
    {
        std::string line;
        for (const auto &_value : value) {
            if (measure == Measure::Average || measure == Measure::AverageTotal || measure == Measure::Special) {
                line += std::to_string(get_average(_value[layer])) + ',';
            } else if (measure == Measure::Total) {
                line += std::to_string(get_total(_value[layer])) + ',';
            } else if (measure == Measure::Max) {
                line += std::to_string(get_max(_value[layer])) + ',';
            } else {
                throw std::runtime_error("Wrong measure formula");
            }
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_double_dist_t::network_to_string()
    {
        std::string line;
        for (const auto &_value : value) {
            if (measure == Measure::Average) {
                line += std::to_string(get_average(_value,skip_first)) + ',';
            } else if (measure == Measure::AverageTotal) {
                line += std::to_string(get_average_total(_value)) + ',';
            }else if (measure == Measure::Total) {
                line += std::to_string(get_total(_value)) + ',';
            } else if (measure == Measure::Max) {
                line += std::to_string(get_max(_value)) + ',';
            } else {
                throw std::runtime_error("Wrong measure formula");
            }
        }
        line = line.substr(0, line.size() - 1);
        return line;
    }

    inline std::string stat_double_dist_t::dist_to_string()
    {
        std::string line;
        for (int64_t r = min_range; r <= max_range; ++r)
            line += std::to_string(r) + ',';
        line = line.substr(0, line.size() - 1);
        return line;
    }

    // Stats

    Stats::Stats(uint64_t _layers, uint64_t _images, const std::string &_filename) : layers(_layers),
            images(_images)
    {
        filename = _filename;
    }

    void Stats::check_path(const std::string &path)
    {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    std::shared_ptr<stat_string_t> Stats::register_string_t(const std::string &name, Measure measure, bool skip_first) {
        table_t table;
        table.name = name;
        table.var = std::make_shared<stat_string_t>(stat_string_t(layers, images, "-", measure, skip_first));

        database.emplace_back(table);
        return std::dynamic_pointer_cast<stat_string_t>(table.var);
    }

    std::shared_ptr<stat_uint_t> Stats::register_uint_t(const std::string &name, uint64_t init_value, Measure measure,
            bool skip_first)
    {
        table_t table;
        table.name = name;
        table.var = std::make_shared<stat_uint_t>(stat_uint_t(layers, images, init_value, measure, skip_first));

        database.emplace_back(table);
        return std::dynamic_pointer_cast<stat_uint_t>(table.var);
    }

    std::shared_ptr<stat_double_t> Stats::register_double_t(const std::string &name, double init_value, Measure measure,
            bool skip_first)
    {
        table_t table;
        table.name = name;
        table.var = std::make_shared<stat_double_t>(stat_double_t(layers, images, init_value, measure, skip_first));

        database.emplace_back(table);
        return std::dynamic_pointer_cast<stat_double_t>(table.var);
    }

    std::shared_ptr<stat_uint_dist_t> Stats::register_uint_dist_t(const std::string &name, int64_t min_range,
            int64_t max_range, uint64_t init_value, Measure measure, bool skip_first)
    {
        table_t table;
        table.name = name;
        table.var = std::make_shared<stat_uint_dist_t>(stat_uint_dist_t(layers, images, min_range, max_range,
                init_value, measure, skip_first));

        database.emplace_back(table);
        return std::dynamic_pointer_cast<stat_uint_dist_t>(table.var);
    }

    std::shared_ptr<stat_double_dist_t> Stats::register_double_dist_t(const std::string &name, int64_t min_range,
            int64_t max_range, double init_value, Measure measure, bool skip_first)
    {
        table_t table;
        table.name = name;
        table.var = std::make_shared<stat_double_dist_t>(stat_double_dist_t(layers, images, min_range, max_range,
                init_value, measure, skip_first));

        database.emplace_back(table);
        return std::dynamic_pointer_cast<stat_double_dist_t>(table.var);
    }

    void Stats::dump_csv(const std::string &network_name, const std::vector<std::string> &layers_name,
            const std::string &header, bool QUIET)
    {

        std::ofstream o_file;

        try {
            check_path("results");
        } catch (const std::exception &exception) {
            if (mkdir("results", 0775) == -1)
                throw std::runtime_error("Error creating folder results");
        }

        try {
            check_path("results/" + network_name);
        } catch (const std::exception &exception) {
            if (mkdir(("results/" + network_name).c_str(), 0775) == -1)
                throw std::runtime_error("Error creating folder results/" + network_name);
        }

        std::string path = "results/" + network_name + "/" + filename + ".csv";
        o_file.open (path);

        o_file << std::endl << header << std::endl;


        bool scalar = false;
        for (const auto &table : database) {
            if (table.var->getType() == stat_type::Scalar)
                scalar = true;
        }

        if (scalar) {

            std::string scalar_parameter_names = "Layer,image,";
            for (const auto &table : database) {
                if (table.var->getType() == stat_type::Scalar)
                    scalar_parameter_names += table.name + ',';
            }
            scalar_parameter_names = scalar_parameter_names.substr(0, scalar_parameter_names.size() - 1);

            o_file << std::endl << "Per image scalar results:" << std::endl;
            o_file << scalar_parameter_names << std::endl;

            for (uint64_t image = 0; image < images; ++image) {
                for (uint64_t layer = 0; layer < layers; ++layer) {

                    std::string line = layers_name[layer] + ',' + std::to_string(image) + ',';
                    for (const auto &table : database) {
                        if (table.var->getType() == stat_type::Scalar)
                            line += table.var->to_string(layer, image) + ',';
                    }
                    line = line.substr(0, line.size() - 1);
                    o_file << line << std::endl;

                }
            }

            o_file << std::endl << "Layer scalar results:" << std::endl;
            o_file << scalar_parameter_names << std::endl;
            for (uint64_t layer = 0; layer < layers; ++layer) {

                std::string line = layers_name[layer] + ",ALL,";
                for (const auto &table : database) {
                    if (table.var->getType() == stat_type::Scalar)
                        line += table.var->layer_to_string(layer) + ',';
                }
                line = line.substr(0, line.size() - 1);
                o_file << line << std::endl;

            }

            o_file << std::endl << "Network scalar results:" << std::endl;
            o_file << scalar_parameter_names << std::endl;
            std::string line = network_name + ",ALL,";
            for (const auto &table : database) {
                if (table.var->getType() == stat_type::Scalar) {
                    if (table.var->measure == Measure::Special)
                        line += std::to_string(table.var->special_value) + ',';
                    else
                        line += table.var->network_to_string() + ',';
                }
            }
            line = line.substr(0, line.size() - 1);
            o_file << line << std::endl;
        }


        for (const auto &table : database) {
            if (table.var->getType() == stat_type::Distribution) {

                std::string parameter_names = "Layer,Image," + table.var->dist_to_string();

                o_file << std::endl << "Per image " << table.name << " results:" << std::endl;
                o_file << parameter_names << std::endl;

                for(uint64_t image = 0; image < images; ++image) {
                    for (uint64_t layer = 0; layer < layers; ++layer) {
                        std::string line = layers_name[layer] + ',' + std::to_string(image) + ',';
                        o_file << line << table.var->to_string(layer, image) << std::endl;
                    }
                }

                o_file << std::endl << "Layer " << table.name << " results:" << std::endl;
                o_file << parameter_names << std::endl;
                for (uint64_t layer = 0; layer < layers; ++layer) {
                    std::string line = layers_name[layer] + ",ALL,";
                    o_file << line << table.var->layer_to_string(layer) << std::endl;
                }

                o_file << std::endl << "Network " << table.name << " results:" << std::endl;
                o_file << parameter_names << std::endl;
                std::string line = network_name + ",ALL,";
                o_file << line << table.var->network_to_string() << std::endl;

            }
        }

        o_file.close();

        if (!QUIET) std::cout << "Results stored in: " << path << std::endl;
    }

}