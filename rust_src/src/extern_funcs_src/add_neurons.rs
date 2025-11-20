use std::{ffi::c_void, u32::MAX};

use crate::{neural_net_src::{
    neuron_src::create_neuron::create_neuron, rand_id_gen::rand_id_gen, 
    types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Add input neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_input_neuron_ext(
    nn_vp: *mut c_void, id_len: usize, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let rand_id: String = rand_id_gen(id_len);
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "input", 
            (*nn_ptr).edge_counter.clone()
        );
        
        (*nn_ptr).add_input_neuron(rand_id, neuron);
    }
}


/// Add hidden neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_hidden_neuron_ext(
    nn_vp: *mut c_void, id_len: usize, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let rand_id: String = rand_id_gen(id_len);
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "output", 
            (*nn_ptr).edge_counter.clone()
        );
        
        (*nn_ptr).add_hidden_neuron(rand_id, neuron);
    }
}

/// Add output neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_output_neuron_ext(
    nn_vp: *mut c_void, id_len: usize, 
    max_edges: usize,
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let rand_id: String = rand_id_gen(id_len);
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "output", 
            (*nn_ptr).edge_counter.clone()
        );
        
        (*nn_ptr).add_output_neuron(rand_id, neuron);
    }
}