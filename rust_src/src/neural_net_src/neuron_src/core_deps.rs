use std::sync::{Arc, Mutex};

use crate::neural_net_src::edge_src::direct_edge::DirectEdge;

/// Struct that contains all attributes required for input,
/// hidden and output neurons.
pub struct NeuronAttr
{
    // Contains Arc references to edges indicating which neurons the current
    // neuron is connected to.
    forward_edges: Vec<Arc<Mutex<DirectEdge>>>,
    backward_edges: Vec<Arc<Mutex<DirectEdge>>>,

    neuron_level: u32,
    neuron_id: String,
}

/// Contains trait methods for input, hidden and output neurons.
pub trait NeuronTrait
{
    fn forward(&mut self); // Forward propagation.
    fn backward(&mut self); // Backward propagation.

    fn add_edge(&mut self, edge: &mut DirectEdge);
    fn remove_edge(&mut self, edge_index: usize);
}