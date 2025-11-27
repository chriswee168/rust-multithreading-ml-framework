import json
import numpy as np

class NeuralNet:
    """
    Defines objects that interface with the neural network from the rust library.
    """
    def __init__(self, rust_backend_funcs: dict, hyper_param_json_path: str):

        self.rust_backend_funcs = rust_backend_funcs

        with open(hyper_param_json_path, "r") as f:
            hyper_params = json.load(f)

        self.n_threads: int = hyper_params["n_threads"]
        self.in_dim: int = hyper_params["in_dim"]
        self.out_dim: int = hyper_params["out_dim"]
        self.n_input_neurons: int = hyper_params["n_input_neurons"]
        self.n_output_neurons: int = hyper_params["n_output_neurons"]
        self.max_edges: int = hyper_params["max_edges"]
        self.max_depth: int = hyper_params["max_depth"]
        self.lr: float = hyper_params["lr"]
        self.return_grads: bool = hyper_params["return_grads"]

        self.add_neuron_rate: float = hyper_params["add_neuron_rate"]
        self.remove_neuron_rate: float = hyper_params["remove_neuron_rate"]
        self.add_io_edge_rate: float = hyper_params["add_io_edge_rate"]
        self.join_neuron_rate: float = hyper_params["join_neuron_rate"]
        self.remove_edge_rate: float = hyper_params["remove_edge_rate"]
        self.edge_param_thresh: float = hyper_params["edge_param_thresh"]

        self.nn_vp = self.rust_backend_funcs["create_nn"]()

        # Initialise the input and output neurons.
        for i in range(self.n_input_neurons):
            neuron_name = f"input_{i}"
            self.rust_backend_funcs["add_input_neuron"](
                self.nn_vp, neuron_name.encode(), self.max_edges
            )

        for i in range(self.n_output_neurons):
            neuron_name = f"output_{i}"
            self.rust_backend_funcs["add_output_neuron"](
                self.nn_vp, neuron_name.encode(), self.max_edges
            )
    
    def spawn_threads(self):
        """
        Initialise the threads to use for propagation of this neural network.
        """
        self.rust_backend_funcs["spawn_threads"](
            self.nn_vp,
            self.n_threads,
            self.lr,
            self.return_grads
        )
        
        # Initialise the neuron buffers that contain the initial input and
        # output neurons when performing propagation.
        self.rust_backend_funcs["init_forward_buffer"](self.nn_vp)
        self.rust_backend_funcs["init_backward_buffer"](self.nn_vp)
    
    def forward(self, array: np.ndarray):
        """
        Performs forward propagation for neural network.
        
        :param array: Input array
        :type array: np.ndarray
        """

        self.rust_backend_funcs["propagate"](self.nn_vp)