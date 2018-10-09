#ifndef DNNSIM_NUMPYLOADER_H
#define DNNSIM_NUMPYLOADER_H

#include <string>
#include <core/Network.h>
#include <cnpy/NumpyArray.h>

namespace loader {

    class NumpyLoader {

    private:

        /* Path to the csv file containing the network architecture */
        std::string folder_path;

    public:

        /* Constructor
         * @param _folder_path  Path to the folder containing weights and activations
         */
        NumpyLoader(const std::string &_folder_path){folder_path = _folder_path;}

        /* Load the weights into initialized given network
         * @param network       Network with the layers already initialized
         */
        void load_weights(core::Network* network);

        /* Load the activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void load_activations(core::Network* network);

        /* Load the output activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void load_output_activations(core::Network* network);
    };

}

#endif //DNNSIM_NUMPYLOADER_H
