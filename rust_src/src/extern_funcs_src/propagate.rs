use std::{mem::forget, os::raw::c_void, sync::{atomic::Ordering, RwLockReadGuard}};

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Sets the propagation type of neural network, either forward or backward.
#[unsafe(no_mangle)]
pub extern "C" fn prop_forward_ext(nn_vp: *mut c_void, is_forward: bool)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        (*nn_ptr).traverse_forward.store(is_forward, Ordering::SeqCst);
    }
}

/// Performs forward and backward propagation.
#[unsafe(no_mangle)]
pub extern "C" fn propagate_ext(nn_vp: *mut c_void) -> *mut f32
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;

        let is_forward: bool = (*nn_ptr).traverse_forward.load(Ordering::SeqCst);

        (*nn_ptr).propagate();

        let vector_guard: RwLockReadGuard<'_, Vec<f32>>;
        if is_forward
        {
            vector_guard = (*nn_ptr).get_output_vec();
        }
        else
        {
            vector_guard = (*nn_ptr).get_input_grad_vec();
        }

        let mut vector: Vec<f32> = vector_guard.clone();
        let vector_ptr: *mut f32 = vector.as_mut_ptr();

        // Prevent vector from being dropped after function call so it can
        // be read before manual free.
        forget(vector);

        return vector_ptr;
    }
}