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
*   Weights, and Inputs activations in a *.npy file using the following format
*   Full network in a Google protobuf format file
*   Tactical schedule in a protobuf format file

## Installation in the aenao cluster
Must use 10.0.0.136 machine (It has protobuf and a newer version of gcc installed). First allow access to internet:

    ssh -D 12345 <username>@10.0.0.254 -Nf
    
Then, clone the repository using proxychains:

    proxychains4 git clone https://github.com/Isakon12/DNNsim
    
Finally, allow access to gcc-7 in CentOS (not necessary to run simulations):

    scl enable devtoolset-7 bash

## Compilation:
Command line compilation. First we need to configure the project:
    
    cmake3 -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake3 --build cmake-build-release/ --target all

## Set up directories

Create folder **models** including a folder for each network. Every network must include:
   *  train_val.prototxt
   *  trace_params.csv (Instead of the prototxt file)
      *  _Header_: \<Layer\>:\<Input layer\*\>:\<Output channels\>:\<Kernel X\>:\<Kernel Y\>:\<Stride\>:\<Padding\>
      *  \* Input layer is optional
   *  conv_params.csv (Instead of the prototxt file and the precision.txt) 
      *  _Header_: \<Network\>:\<Layer\>:\<Type(conv|fc|lstm)\>:\<Output channels\>:\<Weight channels\>:\<Kernel X\>: \\  
                   \<Kernel Y\>:\<Kernel size\>:\<Stride\>:\<Padding\>:\<Precision\>:\<Magnitude (without sign)\>
      *  Weights are generic precision 0:15                
   *  precision.txt (Optional, contain 5 lines as the example, first line is skipped)
        *   If this file does not exist the precisions are 13:2 for activations and 0:15 for weights
   
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

In the case of **training** simulation, every network must include in subdirectories:
   * weights/$LAYER-$EPOCH-$BATCH-w.npy
   * input/$LAYER-$EPOCH-$BATCH-in.npy
   * outGrad/$LAYER-$EPOCH-$BATCH-wGrad.npy
   * outGrad/$LAYER-$EPOCH-$BATCH-inGrad.npy
   * outGrad/$LAYER-$EPOCH-$BATCH-wGrad.npy
       
## Test

Print help:

    ./DNNsim -h
    
The simulator instructions are defined in prototxt files. Example files can be found [here](examples/).

Results from simulations can be found inside the results folder. One csv file for each simulation 
containing one line for each layer which are grouped per batch. After that, one line for the each layer is shown with the 
average results for all batches. Finally, the last line corresponds to the total of the network. 
(In the case of training the results are grouped per epoch)

## Command line options

* Option **--threads <positive_num>** indicates the number of simultaneous threads that can be executed. The code is 
parallelized per batch using OpenMP library
* Option **--quiet** remove stdout messages from simulations.
* Option **--fast_mode** makes the simulation execute only one batch per network, the first one.
* Option **--store_fixed_point_protobuf** store the fixed point network in a intermediate Protobuf file.

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
| None | Special generic architecture | [None](examples/None/README.md) |
| BitPragmatic | **Ae**: Exploits bit-level sparsity of activations | [BitPragmatic](examples/BitPragmatic/README.md) |
| Stripes | **Ap**: Exploits precision requirements of activations | [Stripes](examples/Stripes/README.md) |
| DynamicStripes | **Ap**: Exploits dynamic precision requirements of a group of activations | [DynamicStripes](examples/DynamicStripes/README.md) |
| Loom | **Wp + Ap**: Exploits precision requirements of weights and dynamic group of activations | [Loom](examples/Loom/README.md) |
| Laconic | **We + Ae**: Exploits bit-level sparsity of both weights and activations | [Laconic](examples/Laconic/README.md) |
| BitTacticalP | **W + Ap**: Skips zero weights and exploits precision requirements of activations | [BitTactical](examples/BitTactical/README.md) |
| BitTacticalE | **W + Ae**: Skips zero weights and exploits bit-level sparsity of activations | [BitTactical](examples/BitTactical/README.md) |
| SCNN | **W + A**: Skips zero weights and zero activations | [SCNN](examples/SCNN/README.md) |
| SCNNp | **W + A + Ap**: Skips zero weights, zero activations, and exploits precision requirements of activations | [SCNNp](examples/SCNNx/README.md) | 
| SCNNe | **W + A + Ae**: Skips zero weights, zero activations, and exploits bit-level sparsity of activations | [SCNNe](examples/SCNNx/README.md) |
| BitFusion | **Wp + Ap**: Exploits precision requirements of activations and weights for powers of two | [BitFusion](examples/BitFusion/README.md) |

*  Allowed tasks for these architectures:

| Task | Description | 
|:---:|:---:|
| Cycles | Simulate number of cycles and memory accesses | 
| Potentials | Calculate ideal speedup and work reduction | 
| Schedule | Schedule weights statically (Only for BitTactical architecture) |
| AvgWidth | Calculate average effective width for the activations and weights per group (Only for DynamicStripes architecture) |

* Allowed task for the special architecture "None":

| Task | Description | Data type |
|:---:|:---:|:---:|
| Information | Output information of the network | Fixed16 | 
| Sparsity | Calculate sparsity for actiations and weights, number of zero values | Fixed16, Float32 |
| BitSparsity | Calculate bit sparsity for activations and weights, number of zero bits | Fixed16 |

## Allowed Training simulations

*  Allowed input types for the simulations:

| inputType | Description | 
|:---:|:---:|
| Trace | Load network model from *trace_params.csv*, precisions from *precision.txt*, and traces from numpy arrays | 

*  Allowed architectures:

| Architecture | Description | Details | 
|:---:|:---:|:---:|
| None | Special generic architecture | [None](examples/None/README.md) |
| DynamicStripesFP | **Ap**: Exploits dynamic precision requirements of a group of activations | [DynamicStripesFP](examples/DynamicStripesFP/README.md) |
| DynamicTactical | **A + G**: Skips zero activations and gradients | [DynamicTactical](examples/DynamicTactical/README.md) |

*  Allowed tasks for these architectures:

| Task | Description | 
|:---:|:---:|
| Cycles | Simulate number of cycles and memory accesses | 
| Potentials | Calculate ideal speedup and work reduction | 
| AvgWidth | Calculate average effective width for the activations and weights per group (Only for DynamicStripesFP architecture) |

* Allowed task for the special architecture "None":

| Task | Description | Data type |
|:---:|:---:|:---:|
| Sparsity | Calculate sparsity for forward and backward, number of zero values | BFloat16 |
| ExpBitSparsity | Calculate the bit-sparsity only for the exponent, number of zero bits | BFloat16 |
| MantBitSparsity | Calculate sparsity only for the mantissa, number of zero bits | BFloat16 |
| ExpDistr | Print exponent data distribution for forward and backward | BFloat16 |
| MantDistr | Print mantissa data distribution for forward and backward | BFloat16 |

## Input Parameters Description   

The batch file can be constructed as follows for the simulation tool:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| network | string | Name of the network as in the folder models | Valid path | N/A |
| batch | uint32 | Corresponding batch for the Numpy traces | Positive numbers | 0 | 
| epoch | uint32 | Number of epochs in the Numpy traces | Positive numbers | 1 | 
| model | string | Format of the input model definition and traces | Trace-Caffe-CParams-Protobuf-Gzip | N/A |
| input_type | string | Data type of the input traces | Float32-Fixed16-BFloat16 | N/A |
| network_bits | uint32 | Number of baseline bits of the network | Positive Number | 16 |
| tensorflow_8b | bool | Use tensorflow 8bits quantization | True-False | False |
| training | bool | Change mode to training simulations | True-False | False |
| only_forward | bool | Only forward traces in the training simulations | True-False | False |
| only_backward | bool | Only backward traces in the training simulations | True-False | False |
| decoder_states | uint32 | Number of decoder traces for Seq2Seq simulations in training | Positive numbers | 0 (Not Seq2Seq) |

Experiments for the simulation tool can contain the following parameters.

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| architecture | string | Name of the architecture to simulate | Allowed architectures | N/A |
| task | string | Name of the architecture to simulate | Allowed tasks | N/A |
| | | List of Parameters per Architecture | | |
