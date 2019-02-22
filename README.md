# DNNsim 

### Requeriments
*   Cmake posterior to version 3.10
*   GNU C++ compiler posterior to version 5.1
*   Google Protobuf for C++. Installation link:
    *   https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

### Allowed input files

*   The architecture of the net in a train_val.prototxt file (without weights and activations)
*   The architecture of the net in a trace_params.csv file (without weights and activations)
*   The architecture of the net in a conv_params.csv file (without weights and activations)
*   Weights, Bias, Inputs and outputs activations in a *.npy file using the following format
*   Full network in a Google protobuf format file
*   Full network in a Gzip Google protobuf format
*   Tactical schedule in a protobuf format

### Allowed output files

*   Full network in a Google protobuf format file
*   Full network in a Gzip Google protobuf format
*   Tactical schedule in a protobuf format

### Installation in the aenao cluster
Must use 10.0.0.136 machine (It has protobuf and a newer version of gcc installed). First allow access to the web:

    ssh -D 12345 <username>@10.0.0.254 -Nf
    
Then, clone the repository using proxychains:

    proxychains4 git clone https://github.com/Isakon12/DNNsim
    
Finally, allow access to gcc-7 in CentOS (not necessary to run simulations):

    scl enable devtoolset-7 bash

### Compilation:
Command line compilation. First we need to configure the project:
    
    cmake3 -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake3 --build cmake-build-release/ --target all

### Set up directories

Create folder **models** including a folder for each network. Every network must include:
   *  train_val.prototxt
   *  trace_params.csv (Instead of the prototxt file)
   *  conv_params.csv (Instead of the prototxt file)   
   *  precision.txt (Optional, contain 5 lines as the example, first line is skipped)
        *   If this file does not exist the precisions are 13:2 for activations and 0:15 for weights
   
   ```
   magnitude (+1 of the sign), fraction, wgt_magnitude, wgt_fraction
   9;9;8;9;9;8;6;4;
   -1;-2;-3;-3;-3;-3;-1;0;
   2;1;1;1;1;-3;-4;-1;
   7;8;7;8;8;9;8;8;
   ```
    
Create folder **net_traces** including a folder for each network. Every network must include:
   * wgt-$NAME.npy
   * bias-$NAME.npy (Optional, only for inference)
   * act-$NAME-0.npy
   * act-$NAME-0-out.npy (Optional, only for inference)
   
Create folder **results** including a folder for each network. The corresponding results will appear in this subfolders.
    
### Test

Print help:

    ./cmake-build-release/bin/DNNsim -h
    
The simulator instructions are defined in prototxt files. Examples can be found in folder **examples**.

##### Transform tool example 
Transform example inside folder examples, bvlc_alexnet from float32 caffe to fixed16 protobuf, bvlc_alexnet from caffe
to Gzip including bias and output activations, and bvlc_googlenet from float32 caffe to fixed16 protobuf:

    ./cmake-build-release/bin/DNNsim examples/transform_example

The corresponding output protobufs and gzips are located inside the net_traces folder. Examples:

    bvlc_alexnet-t2.proto (t2 for unsigned 2 bytes)
    bvlc_alexnet-f4.gz (f4 for float32)
    bvlc_googlenet-t2.proto
    
**Important note**: This protobuf and gzip files are only generated the first time this is executed, if want to generate
a new file you need to delete the previous file manually. This may be the case when the precision.txt file is changed.
    

##### Simulator tool example
Inference example inside folder examples, inference for Gzip bvlc_alexnet, and for Caffe bvlc_googlenet:

    ./cmake-build-release/bin/DNNsim examples/inference_example

This calculates manually the output activations for each layer and compares its output with Caffe output activations. 
It prints for each layer of the network a line indicating the number of errors.

Architectures simulation example inside folder examples. Result for bvlc_googlenet: Laconic cycles, Laconic potentials, 
and BitPragmatic potentials. Results for bvlc_alexnet: BitPragmatics cycles using two different configurations.

    ./cmake-build-release/bin/DNNsim examples/simulator_example

Results can be found inside folder bvlc_alexnet and bvlc_googlenet inside results. One csv file for each simulation 
containing one line for each layer and are grouped per batch. After that, one line for the each layer is shown with the 
average results for all batches. Finally, the last line corresponds to the total of the network.

### Additional options

* Option **--threads <positive_num>** indicates the number of simultaneous threads that can be executed. The code is 
parallelized per batch using OpenMP library
* Option **--fast_mode** makes the simulation execute only one batch per network, the first one.

### Notes about simulations
    
*   Missing support for LSTM layers in the accelerators
*   Mobilenet only allowed for SCNN
*   First layers for our group architectures are transformed using channel folding for strides greater than 1

### Allowed simulations

| Architecture | Description | 
|:---:|:---:|
| Inference | Forward propagation |
| Stripes | **Ap**: Exploits precision requirements of activations | 
| DynamicStripes | **Ap**: Exploits dynamic precision requirements of a group of activations | 
| BitPragmatic | **Ae**: Exploits bit-level sparsity of activations |
| Laconic | **We + Ae**: Exploits bit-level sparsity of both weights and activations |
| BitTacticalP | **W + Ap**: Skips zero weights and exploits precision requirements of activations | 
| BitTacticalE | **W + Ae**: Skips zero weights and exploits bit-level sparsity of activations | 
| SCNN | **W + A**: Skips zero weights and zero activations |
| SCNNp | **W + A + Ap**: Skips zero weights, zero activations, and exploits precision requirements of activations |
| SCNNe | **W + A + Ae**: Skips zero weights, zero activations, and exploits bit-level sparsity of activations |

Input parameters are the parameters that can be changed for each architecture in the prototxt batch file (Default values
can be found inside **examples/README**)  
Default parameters are defined in the header of each architecture, they can be changed in the specific file  
Data type indicates the possible data types allowed: Float32 for 4bytes floating point, and Fixed16 for 2bytes integer

| Architecture | Input Parameters | Default Parameters\* | Cycles | Potentials | Data type |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Inference | - | - | - | - | Float32 |
| Stripes | N_COLUMNS, N_ROWS, BITS_PE | FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16 | X | X | Fixed16 |
| DynamicStripes | N_COLUMNS, N_ROWS, PRECISION_GRANULARITY, COLUMN_REGISTERS | FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16 | X | X | Fixed16 |
| BitPragmatic | N_COLUMNS, N_ROWS, BITS_FIRST_STAGE, COLUMN_REGISTERS | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16| X | X | Fixed16 |
| Laconic | N_COLUMNS, N_ROWS | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16 | X | X | Fixed16 |
| BitTacticalP | N_COLUMNS, N_ROWS, LOOKAHEAD_H, LOOKASIDE_D, SEARCH_SHAPE, PRECISION_GRANULARITY, COLUMN_REGISTERS | ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16 | X | X | Fixed16 |
| BitTacticalE | N_COLUMNS, N_ROWS, LOOKAHEAD_H, LOOKASIDE_D, SEARCH_SHAPE, BITS_FIRST_STAGE, COLUMN_REGISTERS | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES 16 | X | X | Fixed16 |
| SCNN | Wt, Ht, I, F, out_acc_size, BANKS | ZERO_COUNT | X | X | Fixed16, Float32 |
| SCNNp | Wt, Ht, I, F, out_acc_size, BANKS | ZERO_COUNT | - | X | Fixed16 |
| SCNNe | Wt, Ht, I, F, out_acc_size, BANKS | BOOTH_ENCODING, ZERO_COUNT | - | X | Fixed16 |

*\*Default features can be removed in their specific header file*

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
    *   Stripes: class for the Stripes accelerator
    *   DynamicStripes: class for the Dynamic-Stripes accelerator
    *   BitPragmatic: class for the Bit-Pragmatic accelerator
    *   Laconic: class for the Laconic accelerator
    *   BitTactical: common class for both BitTactical behaviors
    *   BitTacticalP: class for the Bit-Tactical version p accelerator
    *   BitTacticalE: class for the Bit-Tactical version e accelerator
    *   SCNN: class for the SCNN accelerator and common behaviour for SCNN-like architectures
    *   SCNNp: class for the SCNNe accelerator
    *   SCNNe: class for the SCNNp accelerator
*   **interface**: Folder to interface with input/output operations
    *   NetReader: class to read and load a network using different formats
    *   NetWriter: class to write and dump a network using different formats
    *   StatsWriter: class to dump simulation statistics in different formats
*   **proto**: Folder for protobuf definition
    *   network.proto: Google protobuf definition for the network
    *   caffe.proto: Caffe protobuf definition for Caffe networks
    *   batch.proto: Google protobuf definition for the batch file
    *   schedule.proto: Tactical schedule protobuf definition
*   **scripts**: Folder for supporting python scripts
    *   save_net.py: Create traces for the networks given the models
    
### Fixes TODO
*   Dispatchers?
*   Add LSTM layers
*   Add documentacion

