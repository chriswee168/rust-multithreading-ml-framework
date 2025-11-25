import ctypes as C

def get_ctypes_funcs(lib_path: str) -> dict[str, C._NamedFuncPointer]:
    """
    Create all ctypes function to interface with the
    neural net from rust backend.
    """

    rust_backend_lib = C.cdll.LoadLibrary(lib_path)

    all_funcs: dict[str, C._NamedFuncPointer] = {}

    # Create neural network.
    create_nn = rust_backend_lib.create_nn_ext
    create_nn.restype = C.c_void_p
    all_funcs.update({"create_nn": create_nn})

    return all_funcs