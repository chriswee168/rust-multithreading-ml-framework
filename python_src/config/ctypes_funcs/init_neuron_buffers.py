import ctypes as C
from ctypes import CDLL

def get_neuron_buffer_funcs(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Initializes the initial neuron buffers for the neural network to
    use when performing forward and backpropagation.
    """
    
    # Args: nn_vp
    init_forward_buffer = rust_backend_lib.init_forward_buffers_ext
    init_forward_buffer.argtypes = [C.c_void_p]
    all_funcs.update({"init_forward_buffer": init_forward_buffer})

    # Args: nn_vp
    init_backward_buffer = rust_backend_lib.init_backward_buffers_ext
    init_backward_buffer.argtypes = [C.c_void_p]
    all_funcs.update({"init_backward_buffer": init_backward_buffer})