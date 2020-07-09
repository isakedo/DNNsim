# DNNsim 

## Requirements
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
      *  _Header_: \<Name\>:\<Type(conv|fc|rnn)\>:\<Stride\>:\<Padding\>             
      
They can also include a:      
   *  precision.txt (Contain 5 lines as the example, first line is skipped)
        *   If this file does not exist the layers are quantized using linear quantization
        *   If the network traces are already quantized, use profiled flag
   
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
| CSV | Load network model from *model.csv*, precisions from *precision.txt*, and traces from numpy arrays | 

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
| model | string | Format of the input model definition and traces | Caffe-CSV | N/A |
| data_type | string | Data type of the input traces | Float-Fixed | N/A |
| network | string | Name of the network as in the models folder | Valid path | N/A |
| data_width | uint32 | Number of baseline bits of the network | Positive Number | 16 |
| quantised | bool | True if traces already quantised | True-False | False |

Experiments contain the parameters specifics for the memory system and the architectures. 
The memory system parameters are general for all architectures, while architecture are different per architecture. 
They can be found [here](examples/).

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| architecture | string | Name of the architecture to simulate | Allowed architectures | N/A |
| task | string | Name of the architecture to simulate | Allowed tasks | N/A |
| dataflow | string | Name of the dataflow to simulate | Allowed dataflows | N/A |
| | | **DRAM Parameters** | | |
| cpu_clock_freq | string | Compute frequency | NUM (G&#124;M&#124;K) Hz | 1GHz |
| dram_conf | string | DRAM configuration file in "ini" | Valid file | DDR4-3200 |
| dram_size | string | DRAM off-chip size | NUM \(G&#124;M&#124;Gi&#124;Mi) B | 16GiB |
| dram_width | uint32 | DRAM interface width | Positive number | 64 |
| dram_start_act_address | uint64 | DRAM start activations address | Positive Number | 0x80000000 |
| dram_start_wgt_address | uint64 | DRAM start weight address | Positive Number | 0x00000000 |
| | | **Global Buffer Parameters** | | |
| | | **Change to *act* for activations** | | |
| | | **Change to *wgt* for weights** | | |
| gbuffer_*xxx*_levels | uint32 | Global Buffer hierarchy levels | Positive Number | 1 |
| gbuffer_*xxx*_size | string | Global Buffer On-chip size | NUM \(G&#124;M&#124;K&#124;Gi&#124;Mi&#124;Ki\) B | 1GiB |
| gbuffer_*xxx*_banks | uint32 | Global Buffer On-chip banks | Positive Number | 32/256 |
| gbuffer_*xxx*_bank_width | uint32 | Global Buffer bank interface width in bits | Positive Number | 256 |
| gbuffer_*xxx*_read_delay | uint32 | Global Buffer read delay in cycles | Positive Number | 2 |
| gbuffer_*xxx*_write_delay | uint32 | Global Buffer write delay in cycles | Positive Number | 2 |
| gbuffer_*xxx*_eviction_policy | string | Global Buffer Eviction policy for lower levels | LRU-FIFO | LRU |
| | | **Local Buffer Parameters** | | |
| | | **abuffer for activations** | | |
| | | **wbuffer for weights** | | |
| | | **pbuffer for partial sums** | | |
| | | **obuffer for output activations** | | |
| *x*buffer_rows | uint32 | Activation Buffer memory rows | Positive Number | 2 |
| *x*buffer_read_delay | uint32 | Activation Buffer read delay in cycles | Positive Number | 1 |
| *x*buffer_write_delay | uint32 | Activation Buffer read delay in cycles | Positive Number | 1 |
| | | **Other modules Parameters** | | |
| composer_inputs | uint32 | Composer column parallel inputs per tile | Positive Number | 256 |
| composer_delay | uint32 | Composer column delay | Positive Number | 1 |
| ppu_inputs | uint32 | Post-Processing Unit parallel inputs | Positive Number | 16 |
| ppu_delay | uint32 | Post-Processing Unit delay | Positive Number | 1 |
| | | [**Architecture Parameters**](examples/) | | |
