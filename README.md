# DNNsim 0.3.0

Gitignore is set up for CLion IDE, if you want to use a different IDE add their project file extensions to .gitignore. 
It recommend to used it from the command line. The folder "models" must contain a folder for each network. 
Inside each folder the file "train_val.protoxt" is mandatory, while "precision.txt". This file must have 5 lines
corresponding to the final precisions and the header. It is optional (If not set the precision is generic)
The weights and activations for each network in their corresponding network folder inside the folder "net_traces".  

The tool works with batch files that can be found in the folder "examples". All the options and parameters are described
in the README file in that folder using BNF like notation. The file "common.h" under the folder "sys/include/sys" 
contain global variables. Check this file before launch any simulation.

### Requeriments
*   Cmake posterior to version 3.10
*   Google Protobuf for C++. Installation link:
    *   https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

### Allowed input files

*   The architecture of the net in a train_val.prototxt file (without weights and activations)
*   Weights, Inputs and outputs activations in a *.npy file using the following format:
    *   wgt-$NAME.npy | act-$NAME-0{-out}.npy
*   Full network in a Google protobuf format file
*   Full network in a Gzip Google protobuf format

### Allowed output files

*   Full network in a Google protobuf format file
*   Full network in a Gzip Google protobuf format

### Compilation:
Command line compilation. First we need to configure the project (It can be Debug or Release for optimizations):
    
    cmake -H. -Bcmake-build-debug -DCMAKE_BUILD_TYPE=Debug

Then, we can proceed to build the project

    cmake --build cmake-build-debug/ --target all
    
### Test

Print help:

    ./cmake-build-debug/bin/DNNsim -h

##### Transform tool example 
Transform example inside folder examples, bvlc_alexnet from float32 caffe to fixed16 protobuf, bvlc_alexnet from caffe
to Gzip including bias and output activations, and bvlc_googlenet from float32 caffe to fixed16 protobuf:

    ./cmake-build-debug/bin/DNNsim examples/transform_example

##### Simulator tool example
Inference example inside folder examples, inference for Gzip bvlc_alexnet, and for Caffe bvlc_googlenet:

    ./cmake-build-debug/bin/DNNsim examples/inference_example

Architectures simulation example inside folder examples, BitPragmatic memory accesses for bvlc_alexnet, and Laconic
potentials for bvlc_googlenet:

    ./cmake-build-debug/bin/DNNsim examples/simulator_example

### Structure:
*   **sys**: Folder for system libraries
    *   common: contains common definitions for all classes
    *   cxxopts: library to read options from the console (c) Jarryd Beck
    *   Statistic: container for the statistics of the simulation
    *   Batch: support to load batch files
*   **cnpy**: Folder for supporting math libraries
    *   cnpy: library to read Numpy arrays
    *   Array: class to store and operate with flatten arrays
*   **core**: Folder for the main classes of the simulator
    *   Network: class to store the network
    *   Layer: class to store the layer of the network
    *   InferenceSimulator: class that defines the behaviour of a standard deep learning inference simulation
    *   TimingSimulator: class that define common behaviour for timing simulation
    *   BitPragmatic: class for the Bit-Pragmatic accelerator
    *   Laconic: class for the Laconic accelerator
*   **interface**: Folder to interface with input/output operations
    *   NetReader: class to read and load a network using different formats
    *   NetWriter: class to write and dump a network using different formats
    *   StatsWriter: class to dump simulation statistics in different formats
*   **proto**: Folder for protobuf definition
    * network.proto: Google protobuf definition for the network
    * caffe.proto: Caffe protobuf definition for Caffe networks
    * batch.proto: Google protobuf definition for the batch file
        
### Things to do:
*  Pragmatic
*  Laconic
*  Bit-fusion
*  LSTM
