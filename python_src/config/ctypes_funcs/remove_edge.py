import ctypes as C
from ctypes import CDLL

def get_remove_edge_func(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Function to remove a random edge in the neural network. The edge is only removed if
    its absolute average of its parameters is below a parameter threshold indicating it
    contributes little to the neural network's output.
    """

    # Args: nn_vp, edge_param_thresh
    remove_random_edge = rust_backend_lib.remove_random_edge_ext
    remove_random_edge.argtypes = [C.c_void_p, C.c_float]
    all_funcs.update({"remove_random_edge": remove_random_edge})