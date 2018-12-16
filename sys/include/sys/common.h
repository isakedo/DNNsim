#ifndef DNNSIM_COMMON_H
#define DNNSIM_COMMON_H

#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <sstream>

// Don't commit changes in global variables
#define DEBUG
//#define STOP_AFTER_ERROR
//#define OPENMP
//#define OUTPUT_ACTIVATIONS
//#define BIAS

#define INITIALISE_DATA_TYPES(name) \
    char gInstantiationGuard##name; \
    template class name<float>; \
    template class name<uint16_t>

#endif //DNNSIM_COMMON_H
