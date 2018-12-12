# DNNsim 0.1.5

Gitignore is set up for CLion IDE, if you want to use a different IDE add their project file extensions to .gitignore. 
It recommend to used it from the command line. The folder "models" must contain a folder for each network. 
Inside each folder the file "train_val.protoxt" is mandatory, while "precision.txt" is optional (If not set the precision is generic)
The weights and activations for each network in their corresponding network folder inside the folder "net_traces".

### Things to do:
*  Pragmatic
*  Laconic
*  Bit-fusion
*  Booth enconding
*  LSTM
*  Change options by a configuration file
*  Allow batching

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
For caffe model and Numpy arrays:

    ./cmake-build-debug/bin/DNNsim Transform -n bvlc_alexnet --ditype Float32 --itype Caffe --otype Protobuf

For Protobuf as input and Gzip as output in fixed point format:

    ./cmake-build-debug/bin/DNNsim Transform -n bvlc_alexnet --ditype Float32 --itype Protobuf --odtype Fixed16 --otype Protobuf

##### Simulator tool example

For caffe model and Numpy arrays:

    ./cmake-build-debug/bin/DNNsim Simulator -n bvlc_alexnet --ditype Float32 --itype Caffe

### Structure:
*   **sys**: Folder for system libraries
    *   common: contains common definitions for all classes
    *   cxxopts: library to read options from the console (c) Jarryd Beck
    *   Statistic: container for the statistics of the simulation
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
    * proto: Folder for protobuf definition
        * network.proto Google protobuf definition for the network
        * caffe.proto Caffe protobuf definition for Caffe networks
