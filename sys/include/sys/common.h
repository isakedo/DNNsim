#ifndef DNNSIM_COMMON_H
#define DNNSIM_COMMON_H

#include <cstdlib>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <memory>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <sstream>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <math.h>

// Don't commit changes in global variables
// If want to launch program uncomment DEBUG, and comment OPENMP
//#define DEBUG
#define STOP_AFTER_ERROR
#define OPENMP

#define INITIALISE_DATA_TYPES(name) \
    char gInstantiationGuard##name; \
    template class name<float>; \
    template class name<uint16_t>

#endif //DNNSIM_COMMON_H
