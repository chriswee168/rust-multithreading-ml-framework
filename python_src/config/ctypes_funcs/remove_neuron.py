import ctypes as C
from ctypes import CDLL

def get_remove_neuron_func(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Function to remove a random hidden neuron that has no backward or forward edges.
    """

    # Args: nn_vp
    remove_random_neuron = rust_backend_lib.remove_rand_hidden_neuron_ext
    remove_random_neuron.argtypes = [C.c_void_p]
    all_funcs.update({"remove_random_neuron": remove_random_neuron})