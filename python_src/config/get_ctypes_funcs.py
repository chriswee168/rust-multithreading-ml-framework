import ctypes as C
from python_src.config.ctypes_funcs import *

def get_ctypes_funcs(lib_path: str) -> dict:
    """
    Create all ctypes function to interface with the
    neural net from rust backend.
    """

    rust_backend_lib = C.cdll.LoadLibrary(lib_path)

    all_funcs: dict = {}

    create_nn.get_create_nn_func(rust_backend_lib, all_funcs)
    add_neurons.get_add_neurons_funcs(rust_backend_lib, all_funcs)
    init_neuron_buffers.get_neuron_buffer_funcs(rust_backend_lib, all_funcs)
    add_io_edges.get_add_io_edge_funcs(rust_backend_lib, all_funcs)
    join_neurons.get_join_neurons_func(rust_backend_lib, all_funcs)
    propagate.get_propagation_funcs(rust_backend_lib, all_funcs)
    remove_edge.get_remove_edge_func(rust_backend_lib, all_funcs)
    set_edge_params.get_set_edge_params_func(rust_backend_lib, all_funcs)
    spawn_threads.get_spawn_threads_func(rust_backend_lib, all_funcs)
    display_params.get_display_params_func(rust_backend_lib, all_funcs)

    return all_funcs