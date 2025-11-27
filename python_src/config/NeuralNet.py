import json

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

        self.add_neuron_rate: float = hyper_params["add_neuron_rate"]
        self.remove_neuron_rate: float = hyper_params["remove_neuron_rate"]
        self.add_io_edge_rate: float = hyper_params["add_io_edge_rate"]
        self.join_neuron_rate: float = hyper_params["join_neuron_rate"]
        self.remove_edge_rate: float = hyper_params["remove_edge_rate"]
        self.edge_param_thresh: float = hyper_params["edge_param_thresh"]

        self.nn_vp = self.rust_backend_funcs["create_nn"]()
