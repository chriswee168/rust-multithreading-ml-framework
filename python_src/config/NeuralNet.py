import json
import numpy as np
from copy import deepcopy

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
        self.max_io_edges: int = hyper_params["max_io_edges"]
        self.max_hidden_edges: int = hyper_params["max_hidden_edges"]
        self.max_depth: int = hyper_params["max_depth"]
        self.neuron_id_len: int = hyper_params["neuron_id_len"]

        self.lr: float = hyper_params["lr"]
        self.return_grads: bool = hyper_params["return_grads"]
        self.lowest_param_val: float = hyper_params["lowest_param_val"]
        self.highest_param_val: float = hyper_params["highest_param_val"]

        self.add_neuron_rate: float = hyper_params["add_neuron_rate"]
        self.add_io_edge_rate: float = hyper_params["add_io_edge_rate"]
        self.join_neuron_rate: float = hyper_params["join_neuron_rate"]
        self.edge_param_thresh: float = hyper_params["edge_param_thresh"]

        self.nn_vp = self.rust_backend_funcs["create_nn"]()

        # Initialise the input and output neurons.
        for i in range(self.n_input_neurons):
            neuron_name = f"input_{i}"
            self.rust_backend_funcs["add_input_neuron"](
                self.nn_vp, neuron_name.encode(), self.max_io_edges
            )

        for i in range(self.n_output_neurons):
            neuron_name = f"output_{i}"
            self.rust_backend_funcs["add_output_neuron"](
                self.nn_vp, neuron_name.encode(), self.max_io_edges
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
        self.rust_backend_funcs["prop_forward"](self.nn_vp, True)
        self.rust_backend_funcs["set_input_vec"](self.nn_vp, array, self.in_dim)
        self.rust_backend_funcs["init_output_vec"](self.nn_vp, self.out_dim)
        output_array_ptr = self.rust_backend_funcs["propagate"](self.nn_vp)
        output_array = deepcopy(np.ctypeslib.as_array(output_array_ptr, (self.out_dim,)))
        self.rust_backend_funcs["free_vec"](output_array_ptr, self.out_dim)

        return output_array
    
    def optimize(self, grad_array: np.ndarray):
        """
        Performs backward propagation for neural network and update edge parameters  
        using gradient descent.
        
        :param grad_array: Output gradient array
        :type grad_array: np.ndarray
        """
        self.rust_backend_funcs["prop_forward"](self.nn_vp, False)
        self.rust_backend_funcs["set_output_grad_vec"](self.nn_vp, grad_array, self.out_dim)
        self.rust_backend_funcs["init_input_grad_vec"](self.nn_vp, self.in_dim)
        input_grad_array_ptr = self.rust_backend_funcs["propagate"](self.nn_vp)
        input_grad_array = deepcopy(np.ctypeslib.as_array(input_grad_array_ptr, (self.in_dim,)))
        self.rust_backend_funcs["free_vec"](input_grad_array_ptr, self.in_dim)

        return input_grad_array
    
    def expand(self):
        """
        Performs a mutation that "expands" the neural net to increase its
        complexity, which can include:
        - Adding a new neuron.
        - Adding a new input or output edge for input or output neurons respectively.
        - Joining two existing neurons.
        """

        random_choice = np.random.choice(
            3, 1, p=[
                self.add_neuron_rate, 
                self.add_io_edge_rate, 
                self.join_neuron_rate
            ]
        )[0]
        random_neg_weight = np.random.uniform(self.lowest_param_val, self.highest_param_val)
        random_pos_weight = np.random.uniform(self.lowest_param_val, self.highest_param_val)
        
        # Add a neuron.
        if random_choice == 0:
            random_neg_weight1 = np.random.uniform(self.lowest_param_val, self.highest_param_val)
            random_pos_weight1 = np.random.uniform(self.lowest_param_val, self.highest_param_val)

            random_depth = np.random.randint(0, self.max_depth)
            self.rust_backend_funcs["add_hidden_neuron"](
                self.nn_vp, self.neuron_id_len, self.max_hidden_edges, random_depth,
                random_neg_weight, random_pos_weight,
                random_neg_weight1, random_pos_weight1
            )
        
        # Add an input/output edge.
        elif random_choice == 1:
            io_edge_option = np.random.choice(2, 1)[0]

            # Add input edge.
            if io_edge_option == 0:
                random_arr_idx = np.random.randint(0, self.in_dim)
                random_neuron_idx = np.random.randint(0, self.n_input_neurons)
                random_input_neuron = f"input_{random_neuron_idx}"
                
                self.rust_backend_funcs["add_input_edge"](
                    self.nn_vp, random_input_neuron.encode(), random_arr_idx,
                    random_neg_weight, random_pos_weight
                )
            
            # Add output edge.
            elif io_edge_option == 1:
                random_arr_idx = np.random.randint(0, self.out_dim)
                random_neuron_idx = np.random.randint(0, self.n_output_neurons)
                random_output_neuron = f"output_{random_neuron_idx}"
                
                self.rust_backend_funcs["add_output_edge"](
                    self.nn_vp, random_output_neuron.encode(), random_arr_idx,
                    random_neg_weight, random_pos_weight
                )
        
        # Join two random neurons.
        elif random_choice == 2:
            # Only allows the follow connections:
            # input -> hidden
            # hidden -> hidden
            # hidden -> output
            # input -> output
            neuron_group1 = np.random.randint(0, 3)
            neuron_group2 = np.random.randint(neuron_group1, 3)

            self.rust_backend_funcs["join_two_rand_neurons"](
                self.nn_vp, random_neg_weight, random_pos_weight,
                neuron_group1, neuron_group2
            )
    
    def reduce(self):
        """
        Performs a mutation that "reduces" the neural net to decrease its
        complexity by removing an edge, as well any subsequent "dead ends"
        where propagation ends up at hidden neurons with no edges as a result of 
        removing said edge.
        """

        self.rust_backend_funcs["remove_random_edge"](
            self.nn_vp, self.edge_param_thresh
        )
    
    def modify_random_edge(self):
        """
        Randomly select an edge to randomly modify its parameters.
        """
        rand_pos_weight = np.random.uniform(self.lowest_param_val, self.highest_param_val)
        rand_neg_weight = np.random.uniform(self.lowest_param_val, self.highest_param_val)
        self.rust_backend_funcs["set_rand_edge_params"](
            self.nn_vp, rand_pos_weight, rand_neg_weight
        )
    
    def display_params(self):
        """
        Display all neurons and their edges.
        """
        self.rust_backend_funcs["display_params"](self.nn_vp)