import ctypes as C
from ctypes import CDLL

def get_load_save_funcs(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Functions to load and save parameters of neural networks.
    """
    
    # Args: nn_vp, model_dir_str
    save_model = rust_backend_lib.save_model_ext
    save_model.argtypes = [C.c_void_p, C.POINTER(C.c_char)]
    all_funcs.update({"save_model": save_model})