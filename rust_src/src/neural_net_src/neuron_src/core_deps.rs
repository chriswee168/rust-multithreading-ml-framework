use std::{collections::HashMap, sync::{Arc, Mutex}};

use crate::neural_net_src::types_aliases::ArcEdgeTrait;

/// Struct that contains all attributes required for input,
/// hidden and output neurons.
pub struct NeuronAttr
{
    // Contains Arc references to edges indicating which neurons the current
    // neuron is connected to.
    pub forward_edges: HashMap<String, ArcEdgeTrait>,
    pub backward_edges: HashMap<String, ArcEdgeTrait>,

    // Used to keep track of values being passed between neurons.
    received_sum: f32,

    neuron_level: u32,
}

impl NeuronAttr
{
    pub fn new(
        max_backward_edges: usize, max_forward_edges: usize, 
        neuron_level: u32
    ) -> Self
    {
        return Self 
        {
            forward_edges: HashMap::with_capacity(max_forward_edges), 
            backward_edges: HashMap::with_capacity(max_backward_edges), 
            neuron_level, received_sum: 0.0
        }
    }

    /// Add a forward edge for this neuron to connect to another neuron.
    pub fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait)
    {
        self.forward_edges.insert(edge_id, edge);
    }

    /// Remove a forward edge to disconnect this neuron from another neuron.
    pub fn remove_forward_edge(&mut self, edge_id: String) -> Option<ArcEdgeTrait>
    {
        let removed_edge: Option<ArcEdgeTrait> = self.forward_edges.remove(&edge_id);
        return removed_edge;
    }

    /// Add a backward edge for this neuron to connect to a previous neuron.
    pub fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait)
    {
        self.backward_edges.insert(edge_id, edge);
    }

    /// Remove a backward edge to disconnect this neuron from a previous neuron.
    pub fn remove_backward_edge(&mut self, edge_id: String) -> Option<ArcEdgeTrait>
    {
        let backward_edge: Option<ArcEdgeTrait> = self.backward_edges.remove(&edge_id);
        return backward_edge;
    }

    /// Increment this neuron's received sum.
    pub fn increment_sum(&mut self, value: f32)
    {
        self.received_sum += value;
    }

    /// Obtain the current received sum of this neuron.
    pub fn get_sum(&self) -> f32
    {
        return self.received_sum;
    }

    /// Reset the received sum of this neuron to zero.
    pub fn zero_sum(&mut self)
    {
        self.received_sum = 0.0;
    }
}

/// Contains trait methods for input, hidden and output neurons.
pub trait NeuronTrait
{
    fn forward(&mut self); // Forward propagation.
    fn backward(&mut self); // Backward propagation.

    // Wrapper methods for received_sum attribute in NeuronAttr.
    fn increment_sum(&mut self, value: f32);
    fn get_sum(&self) -> f32;
    fn zero_sum(&mut self);

    fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait);
    fn remove_forward_edge(&mut self, edge_id: String);
    fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait);
    fn remove_backward_edge(&mut self, edge_id: String);
}