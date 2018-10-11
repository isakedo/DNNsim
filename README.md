# DNNsim

#### Compilation:
Command line compilation. First we need to configure the project (It can be Debug or Release for optimizations):
    
    cmake -H. -Bcmake-build-debug -DCMAKE_BUILD_TYPE=Debug

Then, we can proceed to build the project

    cmake --build cmake-build-debug/ --target all

Finally, we can execute it as:

    ./cmake-build-debug/bin/DNNsim

#### Structure:
*   **cnpy**: Folder for supporting math libraries
    *   cnpy: library to read Numpy arrays
    *   NumpyArray: class to store and operate with numpy arrays
*   **core**: Folder for the main classes of the simulator
    *   Network: class to store the network
    *   Layer: common class to store the layer of the network
    *   ConvolutionalLayer: children class for Convolutional layers
    *   FullyConnectedLayer: children class for Fully Connected layers
*   **interface**: Folder to interface with input/output operations
    *   NetReader: class to read and load a network using different formats
    *   NetWriter: class to write and dump a network using different formats
    * proto: Folder for protobuf definition
        * example.proto ...
        

Gitignore is set up for CLion IDE, if you want to use other add their project files to gitignore. 
No problems if the command line is used.

The folder for models under .gitignore is "models"

We need to load the models for (add if I am missing something):
*   The architecture of the net: This is in the file deploy.prototext
*   Inputs and outputs activations: Output from the savenet script in numpy array format. Milos put some under:
    *   /aenao-99/caffe_models/traces/
*   The weights: We can start using the caffe models in weights.caffemodel or get the weights as numpy arrays from savenet

Current python simulator for Bit-Pragmatic is under: 
*   /aenao-99/delmasl1/cnvlutin-PRA/MIsim/functionalSerial.py
*   /aenao-99/delmasl1/cnvlutin-PRA/MIsim/testSystem.py


