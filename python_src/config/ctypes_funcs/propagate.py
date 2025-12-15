import ctypes as C
from ctypes import CDLL
import numpy as np

def get_propagation_funcs(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Functions responsible for the forward and backward propagation.
    """

    # Args: nn_vp, is_forward
    prop_forward = rust_backend_lib.prop_forward_ext
    prop_forward.argtypes = [C.c_void_p, C.c_bool]
    all_funcs.update({"prop_forward": prop_forward})

    # Args: nn_vp
    # Returns: float pointer to the output array or input gradients
    #          depending on propagation mode.
    propagate = rust_backend_lib.propagate_ext
    propagate.argtypes = [C.c_void_p]
    propagate.restype = C.POINTER(C.c_float)
    all_funcs.update({"propagate": propagate})

    # Args: vec_ptr, length
    free_vec = rust_backend_lib.free_vec_ext
    free_vec.argtypes = [C.POINTER(C.c_float), C.c_uint64]
    all_funcs.update({"free_vec": free_vec})

    # Args: nn_vp, in_dim
    init_input_grad_vec = rust_backend_lib.init_input_grad_vec_ext
    init_input_grad_vec.argtypes = [C.c_void_p, C.c_uint64]
    all_funcs.update({"init_input_grad_vec": init_input_grad_vec})

    # Args: nn_vp, out_dim
    init_output_vec = rust_backend_lib.init_output_vec_ext
    init_output_vec.argtypes = [C.c_void_p, C.c_uint64]
    all_funcs.update({"init_output_vec": init_output_vec})

    # Args: nn_vp, input_arr, in_dim
    set_input_vec = rust_backend_lib.set_input_vec_ext
    set_input_vec.argtypes = [C.c_void_p, np.ctypeslib.ndpointer(C.c_float), C.c_uint64]
    all_funcs.update({"set_input_vec": set_input_vec})

    # Args: nn_vp, output_arr, out_dim
    set_output_grad_vec = rust_backend_lib.set_output_grad_vec_ext
    set_output_grad_vec.argtypes = [C.c_void_p, np.ctypeslib.ndpointer(C.c_float), C.c_uint64]
    all_funcs.update({"set_output_grad_vec": set_output_grad_vec})