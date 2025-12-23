use std::ffi::c_void;

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Initialize the forward buffers to contain input neurons.
#[unsafe(no_mangle)]
pub extern "C" fn init_forward_buffers_ext(nn_vp: *mut c_void)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        (*nn_ptr).init_forward_buffer();
    }
}

/// Initialize the backward buffers to contain output neurons.
#[unsafe(no_mangle)]
pub extern "C" fn init_backward_buffers_ext(nn_vp: *mut c_void)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        (*nn_ptr).init_backward_buffer();
    }
}