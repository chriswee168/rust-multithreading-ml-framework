import ctypes as C
from ctypes import CDLL

def get_add_neurons_funcs(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Add input, hidden and output neurons.  
    """

    # Args: nn_vp, neuron_id, max_edges
    add_input_neuron = rust_backend_lib.add_input_neuron_ext
    add_input_neuron.argtypes = [C.c_void_p, C.POINTER(C.c_char), C.c_uint64]
    all_funcs.update({"add_input_neuron": add_input_neuron})

    # Args: nn_vp, id_len, max_edges, neuron_level (for hidden neuron)
    add_hidden_neuron = rust_backend_lib.add_hidden_neuron_ext
    add_hidden_neuron.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint32]
    all_funcs.update({"add_hidden_neuron": add_hidden_neuron})

    # Args: nn_vp, neuron_id, max_edges
    add_output_neuron = rust_backend_lib.add_output_neuron_ext
    add_output_neuron.argtypes = [C.c_void_p, C.POINTER(C.c_char), C.c_uint64]
    all_funcs.update({"add_output_neuron": add_output_neuron})