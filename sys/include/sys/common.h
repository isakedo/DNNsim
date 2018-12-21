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
#include <bitset>

// Don't commit changes in global variables
// If want to launch simulations comment DEBUG and STOP_AFTER_ERROR, and uncomment OPENMP
#define DEBUG
#define STOP_AFTER_ERROR
//#define OPENMP

#define INITIALISE_DATA_TYPES(name) \
    char gInstantiationGuard##name; \
    template class name<float>; \
    template class name<uint16_t>

#endif //DNNSIM_COMMON_H
