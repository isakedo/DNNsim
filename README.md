# DNNsim 

## Requeriments
*   Cmake version >= 3.10
*   GNU C++ compiler version >= 5.1
*   Google Protobuf for C++. Installation link:
    *   https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
## Allowed input files

*   The architecture of the net in a train_val.prototxt file (without weights and activations)
*   The architecture of the net in a trace_params.csv file (without weights and activations)
*   The architecture of the net in a conv_params.csv file (without weights and activations)
*   Weights, and Inputs activations in a *.npy file
*   Full network in a Google protobuf format file

## Compilation:
Command line compilation. First we need to configure the project:
    
    cmake -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake --build cmake-build-release/ --target all

## Set up directories

Create folder **models** including a folder for each network. Every network must include one of these files:
   *  train_val.prototxt
   *  model.csv (Instead of the prototxt file)
      *  _Header_: \<Name\>:\<Type(conv|fc|lstm)\>:\<Stride\>:\<Padding\>             
      
They can also include a:      
   *  precision.txt (Contain 5 lines as the example, first line is skipped)
        *   If this file does not exist the precisions are 14:2 for activations and 1:15 for weights
        *   If the network traces are already quantized, create precision file with precision 8:0 (for 8b traces)
   
   ```
   magnitude (+1 of the sign), fraction, wgt_magnitude, wgt_fraction
   9;9;8;9;9;8;6;4;
   -1;-2;-3;-3;-3;-3;-1;0;
   2;1;1;1;1;-3;-4;-1;
   7;8;7;8;8;9;8;8;
   ```
    
Create folder **net_traces** including a folder for each network. 
In the case of **inference** simulation, every network must include:
   * wgt-$LAYER.npy
   * act-$LAYER-$BATCH.npy
       
All traces must be in **float32**       
       
## Test

Print help:

    ./DNNsim -h
    
The simulator instructions are defined in prototxt files. Example files can be found [here](examples/).

Results from simulations can be found inside the results folder. One csv file for each simulation 
containing one line for each layer which are grouped per images. After that, one line for the each layer is shown with the 
average results for all images. Finally, the last line corresponds to the total of the network. 

## Command line options

* Option **--quiet** remove stdout messages from simulations.
* Option **--fast_mode** makes the simulation execute only one batch per network, the first one.
* Option **--check_values** calculate the output values and check their correctness.

## Allowed Inference simulations

*  Allowed model types for the simulations:

| model | Description |
|:---:|:---:|
| Caffe | Load network model from *train_val.prototxt*, precisions from *precision.txt*, and traces from numpy arrays |
| Trace | Load network model from *trace_params.csv*, precisions from *precision.txt*, and traces from numpy arrays | 
| CParams | Load network model and precisions from *conv_params.csv*, and traces from numpy arrays | 
| Protobuf | Load network model, precisions, and traces from a protobuf file |

*  Allowed architectures for the experiments:

| Architecture | Description | Details | 
|:---:|:---:|:---:|
| DaDianNao | Baseline DaDianNao machine | [DaDianNao](examples/DaDianNao/README.md) |
| Stripes | **Ap**: Exploits precision requirements of activations | [Stripes](examples/Stripes/README.md) |
| ShapeShifter | **Ap**: Exploits dynamic precision requirements of a group of activations | [ShapeShifter](examples/ShapeShifter/README.md) |
| Loom | **Wp + Ap**: Exploits precision requirements of weights and dynamic group of activations | [Loom](examples/Loom/README.md) |
| BitPragmatic | **Ae**: Exploits bit-level sparsity of activations | [BitPragmatic](examples/BitPragmatic/README.md) |
| Laconic | **We + Ae**: Exploits bit-level sparsity of both weights and activations | [Laconic](examples/Laconic/README.md) |
| SCNN | **W + A**: Skips zero weights and zero activations | [SCNN](examples/SCNN/README.md) |

*  Allowed tasks for these architectures:

| Task | Description | 
|:---:|:---:|
| Cycles | Simulate number of cycles and memory accesses | 
| Potentials | Calculate ideal speedup and work reduction | 

## Input Parameters Description   

The batch file can be constructed as follows for the simulation tool:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| batch | uint32 | Corresponding batch for the Numpy traces | Positive numbers | 0 | 
| epoch | uint32 | Number of epochs in the Numpy traces | Positive numbers | 1 | 
| model | string | Format of the input model definition and traces | Trace-Caffe-CParams-Protobuf | N/A |
| data_type | string | Data type of the input traces | Float32-Fixed16-BFloat16 | N/A |
| network | string | Name of the network as in the models folder | Valid path | N/A |
| network_bits | uint32 | Number of baseline bits of the network | Positive Number | 16 |
| tensorflow_8b | bool | Use tensorflow 8bits quantization | True-False | False |

Experiments contain the following parameters.

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| architecture | string | Name of the architecture to simulate | Allowed architectures | N/A |
| task | string | Name of the architecture to simulate | Allowed tasks | N/A |
| | | **Architecture Parameters** | | |
