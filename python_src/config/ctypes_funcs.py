import ctypes as C

def get_ctypes_funcs(lib_path: str) -> dict[str, C._NamedFuncPointer]:
    """
    Create all ctypes function to interface with the
    neural net from rust backend.
    """

    rust_backend_lib = C.cdll.LoadLibrary(lib_path)

    all_funcs: dict[str, C._NamedFuncPointer] = {}

    return all_funcs