use std::ffi::{c_char, c_void, CStr};

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

#[unsafe(no_mangle)]
pub fn join_neurons_ext(
    nn_vp: *mut c_void, neuron_id1: *mut c_char, neuron_id2: *mut c_char,
    neg_weight: f32, pos_weight: f32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;

        let neuron_id1_str: &str = CStr::from_ptr(neuron_id1).to_str().unwrap();
        let neuron_id2_str: &str = CStr::from_ptr(neuron_id2).to_str().unwrap();
        let edge_id: String = neuron_id1_str.to_string() + "_" + neuron_id2_str;

        (*nn_ptr).join_neurons(
            neuron_id1_str, neuron_id2_str, 
            edge_id, neg_weight, pos_weight
        );
        
    }
}