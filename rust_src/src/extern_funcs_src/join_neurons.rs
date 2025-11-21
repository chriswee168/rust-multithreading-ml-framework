use std::ffi::{c_char, c_void, CStr};

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

/// Join two neurons together using a hidden edge.
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

/// Add an input edge to an input neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_input_edge_ext(
    nn_vp: *mut c_void, input_neuron_id: *mut c_char, arr_idx: usize,
    neg_weight: f32, pos_weight: f32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let neuron_id_str: &str = CStr::from_ptr(input_neuron_id).to_str().unwrap();
        let edge_id: String = arr_idx.to_string() + "_" + neuron_id_str;

        (*nn_ptr).add_input_edge(
            neuron_id_str, edge_id, 
            arr_idx, neg_weight, pos_weight
        );
    }
}


/// Add an output edge to an output neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_output_edge_ext(
    nn_vp: *mut c_void, output_neuron_id: *mut c_char, arr_idx: usize,
    neg_weight: f32, pos_weight: f32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let neuron_id_str: &str = CStr::from_ptr(output_neuron_id).to_str().unwrap();
        let edge_id: String = neuron_id_str.to_string() + "_" + arr_idx.to_string().as_str();

        (*nn_ptr).add_output_edge(
            neuron_id_str, edge_id, 
            arr_idx, neg_weight, pos_weight
        );
    }
}