use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::core_deps::{EdgeAttr, EdgeTrait}, neuron_src::core_deps::NeuronTrait};

/// Connects an index of the input array with an input neuron.
pub struct InputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    input_array_index: usize,
    next_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>
}

// Implement constructor.
impl InputEdge
{
    pub fn new(
        input_array_index: usize, 
        next_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>, 
        weight_range: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(weight_range);
        return Self
        {
            attr: edge_attr,
            input_array_index,
            next_neuron
        };        
    }
}

}