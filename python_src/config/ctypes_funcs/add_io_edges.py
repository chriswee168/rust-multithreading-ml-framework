import ctypes as C
from ctypes import CDLL

def get_add_io_edge_funcs(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Functions to add input and output edges to input and output neurons respectively.
    """

    # Args: nn_vp, input_neuron_id, input_arr_idx, neg_weight, pos_weight
    add_input_edge = rust_backend_lib.add_input_edge_ext
    add_input_edge.argtypes = [C.c_void_p, C.POINTER(C.c_char), C.c_uint64, C.c_float, C.c_float]
    all_funcs.update({"add_input_edge": add_input_edge})

    # Args: nn_vp, output_neuron_id, output_arr_idx, neg_weight, pos_weight
    add_output_edge = rust_backend_lib.add_output_edge_ext
    add_output_edge.argtypes = [C.c_void_p, C.POINTER(C.c_char), C.c_uint64, C.c_float, C.c_float]
    all_funcs.update({"add_output_edge": add_output_edge})