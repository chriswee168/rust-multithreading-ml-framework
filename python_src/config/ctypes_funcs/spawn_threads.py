import ctypes as C
from ctypes import CDLL

def get_spawn_threads_func(rust_backend_lib: CDLL, all_funcs: dict):
    """
    Function to spawn threads for parallel propagation in neural net via Breadth
    First Search.
    """

    # Args: nn_vp, num_threads, lr, return_grad
    spawn_threads = rust_backend_lib.spawn_threads_ext
    spawn_threads.argtypes = [C.c_void_p, C.c_uint64, C.c_float, C.c_bool]
    all_funcs.update({"spawn_threads": spawn_threads})