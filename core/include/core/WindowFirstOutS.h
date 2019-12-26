#ifndef DNNSIM_WINDOWFIRSTOUTS_H
#define DNNSIM_WINDOWFIRSTOUTS_H

#include "OutputStationary.h"

namespace core {

    /**
     * Window first output stationary dataflow
     * @tparam T Data type values
     */
    template <typename T>
    class WindowFirstOutS : public OutputStationary<T> {

    private:

        /**
         * Return name for the dataflow
         * @return Name
         */
        std::string name() override;

        /**
         * Return if schedule the weight buffer
         * @param tile_data
         * @return True if weight buffer to schedule, False if not
         */
        bool next_dataflow_step(std::vector<TileData<T>> &tile_data) override;

    public:

        /**
         * Constructor
         * @param _scheduler
         */
        explicit WindowFirstOutS(const BitTactical<T> &_scheduler) : OutputStationary<T>(_scheduler) {}

    };

}

#endif //DNNSIM_WINDOWFIRSTOUTS_H
