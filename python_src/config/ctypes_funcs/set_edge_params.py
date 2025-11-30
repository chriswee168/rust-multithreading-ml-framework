import ctypes as C
from ctypes import CDLL

def get_set_edge_params_func(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Function to change the positive and negative parameters of a random edge.
    """

    # Args: nn_vp, pos_param, neg_param
    set_rand_edge_params = rust_backend_lib.set_rand_edge_params_ext
    set_rand_edge_params.argtypes = [C.c_void_p, C.c_float, C.c_float]
    all_funcs.update({"set_rand_edge_params": set_rand_edge_params})