import ctypes as C
from python_src.config.ctypes_funcs.add_neurons import get_add_neurons_funcs
from python_src.config.ctypes_funcs.create_nn import get_create_nn_func

def get_ctypes_funcs(lib_path: str) -> dict[str, C._NamedFuncPointer]:
    """
    Create all ctypes function to interface with the
    neural net from rust backend.
    """

    rust_backend_lib = C.cdll.LoadLibrary(lib_path)

    all_funcs: dict[str, C._NamedFuncPointer] = {}

    get_create_nn_func(rust_backend_lib, all_funcs)
    get_add_neurons_funcs(rust_backend_lib, all_funcs)

    return all_funcs