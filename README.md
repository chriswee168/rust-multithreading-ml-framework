# Dynamic/Adaptive AI Model Project
Repository for experimental neural network models that are capable of dynamically rewiring their topologies during their training process. This project uses both a Rust backend to handle the high performance multi-threading needed for model training/inference, and a Python frontend for convenient interfacing with Rust functions to create model training scripts.

List of Contents:
1. [Dependencies](#dependencies)
2. [Project Building](#project-building)
3. [Neural Network Architecture](#neural-network-architecture)
    - [Neuron Edge Architecture](#neuron-edge-architecture)
    - [Neural Network Mutation](#neural-network-mutation)
4. [Training And Testing](#training-and-testing)
    - [Main Test Script](#main-test-script)
    - [Hyperparameters](#hyperparameters)
    - [Saving/Loading Neural Networks](#savingloading-neural-networks)
        - [Saving parameters](#saving-parameters)
        - [Loading parameters](#loading-parameters)
5. [Acknowledgements](#acknowledgements)

## Dependencies

### Python 3.11.0
Third party Python libraries required:
- gymnasium==1.1.1

Third party Python libraries can be installed by running:  
`pip install -r requirements.txt`

### Rust 1.90.0
Rust crates required (crates are already included in Cargo.toml):
- serde = { version = "1.0", features = ["derive"] }
- serde_json = "1.0"
- rand = "0.8.5"

Rust and its dependencies can be installed for either Windows or Linux by following the instructions from: https://rust-lang.org/tools/install/.  
After Rust is installed, version 1.90.0 can be installed by running the following commands
in terminal:
```
# Install Rust version 1.90.0.
rustup install 1.90.0

# Set Rust version to 1.90.0.
rustup override set 1.90.0
```

## Project Building
Neural networks utilise a shared Rust library which handles multi-threading required to optimise training and inference. To build it run the command below in the `rust_src/` directory:
```
cargo build --release
```

This will produce a shared library in the `rust_src/target/release/` directory, named either `ai_core.dll` or `libai_core.so` depending on whether operation system is Windows or Linux respectively.

## Neural Network Architecture
Neural networks in this project differ from the classical feedforward neural networks where data is sequentially passed through distinct layers. In this project models have no distinct hidden layers and neurons can be connected irregularly in an acyclic manner, with some neuron paths from the input to output being longer than others. Due to irregularity, models can't effectively take advantage of GPU parallelism and must rely on high performance CPU multi-threading to perform parallel Breadth First Search for propagation.

### Neuron Edge Architecture
The edges that connect neurons each consist of a pair of weights, only one of the weights are used depending on whether the input value is below or above/equal to zero. Below is pseudocode for a neuron edge:
```
edge(x, w1, w2):
    if x >= 0:
        return x * w1
    else:
        return x * w2 
```
This design aims to inject nonlinearity into the edges which outnumber neurons to compensate for the sparse connectivity of neural networks. The idea of moving nonlinearity to weights is inspired by Kolmogorov Arnold Networks (KANs) that uses B-Splines instead which are more computationally expensive than a simple sign check. The paper to KANs are referenced under [Acknowledgements](#acknowledgements).

### Neural Network Mutation
Models can also dynamically alter their toplogies during training to either grow in complexity by adding new neurons and joining random neurons together with random parameterised connections, or reduce complexity and optimize memory by removing redundant connections between neurons that have a parameter average below a specified threshold or magnitude, typically a small value.

## Training And Testing
### Main Test Script
This project uses the CartPole-v1 environment from the Gymnasium Python library developed by
the Farama Foundation to train and test the performance of the neural networks using a simple reinforcement learning algorithm to obtain the best reward. More details for this library can be found under [Acknowledgements](#acknowledgements).

The cartpole environment test can be started by executing the command below in the project's root directory:
```
python -m python_src.main_gym_test
```

### Hyperparameters
JSON files stored in the `python_src/hyper_params` directory are used to define hyperparameters for neural networks. Each file must include the following hyperparameters enclosed in curly brackets:
- **n_threads**: Number of threads to use for parallel Breadth First Search traversal/propagation through neural network. (Should ideally be kept below the number of cores CPU has to optimize parallelism and reduce context switching.)
- **in_dim**: Dimension of the input array.
- **out_dim**: Dimension of the output array.
- **n_input_neurons**: Number of neurons that read from the input array.
- **n_output_neurons**: Number of neurons that write to the output array.
- **max_io_edges**: Maximum number of edges input and output neurons can have to read from or write to input and output arrays respectively.
- **max_hidden_edges**: Maximum number of forward and backward edges for hidden neurons.
- **max_depth**: Controls how deep neural network can be, neurons can only connect to others that are marked at a lower depth to ensure acyclic connections.
- **neuron_id_len**: How many characters do random ID string have.
- **lr**: Learning rate (neural networks use RMSprop optimizer.)
- **return_grads**: Chose whether to return the gradients for input array.
- **lowest_param_val**: Low bound for random edge parameter initialization.
- **highest_param_val**: High bound for random edge parameter initialization.
- **add_neuron_rate**: Chance of adding a new random neuron when calling `NeuralNet.expand()`.
- **add_io_edge_rate**: Chance of adding a new IO edge to a random input or output neuron when calling `NeuralNet.expand()`.
- **join_neuron_rate**: Change of joining two existing neurons when calling `NeuralNet.expand()`.
- **edge_param_thresh**: Edge parameter threshold to determine whether an edge should be removed. (An edge is determined to be redundant if the magnitude of the parameter average is below the threshold.)

A JSON hyperparameter file for the cartpole environment is already included in `python_src/hyper_params/cartpole-v1.json`.

### Saving/Loading Neural Networks
#### Saving parameters
The parameters of neural nets can be saved to JSON files and loaded later to resume training/inference.
To save neural network parameters, call the `NeuralNet.save(model_dir)` method, where `model_dir` is a path to a directory
or folder to store the parameters in several JSON files.

#### Loading parameters
Empty neural networks can also be initialised by loading parameters from an existing model directory. Below is an example:
```
hyper_params = "path/to/hyperparameters/file"
model_dir = "path/to/model/dir"
model = NeuralNet(hyper_params)
model.load(model_dir)
```

## Acknowledgements

Python libraries used:
- [Gymnasium](https://gymnasium.farama.org/) — Python library that provides environments for training and testing AI models using reinforcement learning techniques. Used to evaluate the performance of dynamic neural nets for this project.
    - License type: MIT
    - Link: https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE

Rust crates used:
- [Rand](https://github.com/rust-random/rand) — Required for generating random parameter values for neuron edges between the low and high bounds specified in JSON hyperparameter files.
    - License type: MIT
    - Link: https://github.com/rust-random/rand/blob/master/LICENSE-MIT

Papers referenced:
- The architectures of neural networks in this project take partial inspiration from Kolmogorov Arnold Networks, introduced in the paper "KAN: Kolmogorov-Arnold Networks", reference below:  
Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold Networks*. ArXiv.org. https://arxiv.org/abs/2404.19756