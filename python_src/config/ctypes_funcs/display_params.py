import ctypes as C
from ctypes import CDLL

def get_display_params_func(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Function to display the parameters of neural network.
    """

    # Args: nn_vp
    display_params = rust_backend_lib.display_params_ext
    display_params.argtypes = [C.c_void_p]
    all_funcs.update({"display_params": display_params})