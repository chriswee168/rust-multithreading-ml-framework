use std::ffi::c_void;

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Call method to display all neurons and their edges.
#[unsafe(no_mangle)]
pub extern "C" fn display_params_ext(nn_vp: *mut c_void)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        (*nn_ptr).display_params();
    }
}