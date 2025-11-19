use std::ffi::c_void;

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Create an empty neural network and return void pointer.
#[unsafe(no_mangle)]
pub extern "C" fn create_nn() -> *mut c_void
{
    let nn: NeuralNetWrapper = NeuralNetWrapper::new();
    let nn_box_ptr: Box<NeuralNetWrapper> = Box::new(nn);
    let nn_raw_ptr: *mut NeuralNetWrapper = Box::into_raw(nn_box_ptr);
    return nn_raw_ptr as *mut c_void;
}