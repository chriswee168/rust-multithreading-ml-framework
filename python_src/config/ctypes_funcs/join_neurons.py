import ctypes as C
from ctypes import CDLL

def get_join_neurons_func(rust_backend_lib: CDLL, all_funcs: dict[str, C._NamedFuncPointer]):
    """
    Function to join two random neurons.
    """
    
    # Args: nn_vp, neg_weight, pos_weight, neuron_group1, neuron_group2
    join_two_rand_neurons = rust_backend_lib.join_two_rand_neurons_ext
    join_two_rand_neurons.argtypes = [C.c_void_p, C.c_float, C.c_float, C.c_uint64, C.c_uint64]
    all_funcs.update({"join_two_rand_neurons": join_two_rand_neurons})