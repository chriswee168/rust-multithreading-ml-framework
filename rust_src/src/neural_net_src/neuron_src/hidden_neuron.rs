use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::{NeuronAttr, NeuronTrait}};

/// Struct to define input neuron attributes and behaviour.
pub struct HiddenNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl HiddenNeuron
{
    pub fn new(
        max_backward_edges: usize, max_forward_edges: usize, 
        neuron_level: u32, neuron_id: String
    ) -> Self
    {
        let neuron_attrs: NeuronAttr = NeuronAttr::new(
            max_backward_edges, max_forward_edges, 
            neuron_level, neuron_id
        );

        return Self
        {
            attr: neuron_attrs,
        }
    }
}
