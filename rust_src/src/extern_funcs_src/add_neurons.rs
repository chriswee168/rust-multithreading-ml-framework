use std::{ffi::{c_char, c_void, CStr}, u32::MAX};

use crate::{neural_net_src::{
    neuron_src::create_neuron::create_neuron, 
    types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Add input neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_input_neuron_ext(
    nn_vp: *mut c_void, id: *mut c_char, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let id_str: String = CStr::from_ptr(id).to_str().unwrap().to_string();
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "input", 0
        );
        
        (*nn_ptr).add_input_neuron(id_str, neuron);
    }
}


/// Add hidden neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_hidden_neuron_ext(
    nn_vp: *mut c_void, id: *mut c_char, 
    max_edges: usize, neuron_level: u32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let id_str: String = CStr::from_ptr(id).to_str().unwrap().to_string();
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "hidden", neuron_level
        );
        
        (*nn_ptr).add_hidden_neuron(id_str, neuron);
    }
}

/// Add output neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_output_neuron_ext(
    nn_vp: *mut c_void, id: *mut c_char, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let id_str: String = CStr::from_ptr(id).to_str().unwrap().to_string();
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "output", MAX
        );
        
        (*nn_ptr).add_output_neuron(id_str, neuron);
    }
}