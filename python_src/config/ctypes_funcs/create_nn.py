import ctypes as C
from ctypes import CDLL

def get_create_nn_func(rust_backend_lib: CDLL, all_funcs: dict[str, C._NamedFuncPointer]):
    """
    Create a new neural network.
    """
    
    # Return: void pointer of neural net
    create_nn = rust_backend_lib.create_nn_ext
    create_nn.restype = C.c_void_p
    all_funcs.update({"create_nn": create_nn})