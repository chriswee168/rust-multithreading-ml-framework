use std::sync::{Arc, Condvar, Mutex};

use crate::neural_net_src::{
    neuron_src::{core_deps::NeuronTrait, hidden_neuron::HiddenNeuron, 
        input_neuron::InputNeuron, output_neuron::OutputNeuron}, 
        types_aliases::ArcNeuronTrait};

/// Create input/hidden/output neuron.
pub fn create_neuron(
    neuron_id: String, max_backward_edges: usize, max_forward_edges: usize,
    neuron_type: &str, neuron_level: u32
) -> ArcNeuronTrait
{
    let neuron: Box<dyn NeuronTrait>;
    if neuron_type == "input"
    {
        neuron = Box::new(
            InputNeuron::new(neuron_id, max_backward_edges, max_forward_edges)
        );
    }
    else if neuron_type == "output" 
    {
        neuron = Box::new(
            OutputNeuron::new(neuron_id, max_backward_edges, max_forward_edges)
        );
    }
    else if neuron_type == "hidden" 
    {
        neuron = Box::new(
            HiddenNeuron::new(neuron_id, max_backward_edges, max_forward_edges, neuron_level)
        );
    }
    else
    {
        panic!("Neuron type {} not accepted.", neuron_type);
    }
    
    // Create neuron arc.
    let neuron: ArcNeuronTrait = Arc::new(Mutex::new(neuron));

    return neuron;
}