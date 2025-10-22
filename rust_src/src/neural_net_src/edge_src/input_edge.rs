use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::core_deps::EdgeAttr, neuron_src::core_deps::NeuronTrait};

/// Connects an index of the input array with an input neuron.
pub struct InputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    input_array_index: usize,
    next_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>
}