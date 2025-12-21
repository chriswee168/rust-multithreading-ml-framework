# Dynamic/Adaptive AI Model Project
Repository for experimental neural network models that are capable of dynamically rewiring their topologies during their training process. This project uses both a Rust backend to handle the high performance multi-threading needed for model training/inference, and a Python frontend for convenient interfacing with Rust functions to create model training scripts.

List of Contents:
1. [Dependencies](#dependencies)
2. [Project Building](#project-building)
3. [Training And Testing](#training-and-testing)
    - [Main Test Script](#main-test-script)
    - [Neural Network Mutation](#neural-network-mutation)
5. [Acknowledgements](#acknowledgements)

## Dependencies

### Python 3.11.0
Third party Python libraries required:
- gymnasium==1.1.1

Third party Python libraries can be installed by running:  
`pip install -r requirements.txt`

### Rust 1.90.0
Rust crates required (crates are already included in Cargo.toml):
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

This will produce a shared library in the `rust_src/target/release/` directory, named either `core.dll` or `libcore.so` depending on whether operation system is Windows or Linux respectively.

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
- **n_threads**: Number of threads to use for propagation through neural network.
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

### Neural Network Mutation
Neural networks can dynamically alter their toplogies during training to either grow in complexity by adding new neurons and joining random neurons together with random parameterised connections, or reduce complexity and optimize memory by removing redundant connections between neurons that have a parameter average below a specified threshold or magnitude, typically a small value.

## Acknowledgements
