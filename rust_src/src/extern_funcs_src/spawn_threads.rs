use std::ffi::c_void;

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Spawn number of threads for forward and backward propagation.
#[unsafe(no_mangle)]
pub extern "C" fn spawn_threads_ext(
    nn_vp: *mut c_void, num_threads: usize, lr: f32, return_grad: bool
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        (*nn_ptr).spawn_threads(num_threads, lr, return_grad);
    }
}