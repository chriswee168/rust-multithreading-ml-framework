use std::{collections::HashMap, sync::{Arc, Condvar, Mutex, RwLockWriteGuard}};

use crate::neural_net_src::{neuron_src::{backward::input_backward, core_deps::{NeuronAttr, NeuronTrait}, forward::{hidden_forward, input_forward}}, types_aliases::{ArcEdgeTrait, NeuronBuffer}};

/// Struct to define input neuron attributes and behaviour.
pub struct InputNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl InputNeuron
{
    pub fn new(
        max_backward_edges: usize, max_forward_edges: usize, 
        neuron_level: u32
    ) -> Self
    {
        let neuron_attrs: NeuronAttr = NeuronAttr::new(
            max_backward_edges, max_forward_edges, 
            neuron_level
        );

        return Self
        {
            attr: neuron_attrs,
        }
    }
}

impl NeuronTrait for InputNeuron
{
    /// Add an edge for this neuron to connect to another neuron.
    fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) 
    {
        self.attr.add_forward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from another neuron.
    fn remove_forward_edge(&mut self, edge_id: &str) 
    {
        self.attr.remove_forward_edge(edge_id);
    }
    /// Add an edge for this neuron to connect to a previous neuron.
    fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) 
    {
        self.attr.add_backward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from a previous neuron.
    fn remove_backward_edge(&mut self, edge_id: &str) 
    {
        self.attr.remove_backward_edge(edge_id);
    }
    /// Perform forward pass.
    fn forward(&mut self, neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>) 
    {
        // Obtain the dot products of edges with the input array.
        input_forward(&mut self.attr);

        // Propagate values to hidden neurons.
        hidden_forward(&mut self.attr, neuron_buffer);
    }
    fn backward(
        &mut self, 
        lr: f32, edge_counter: &Arc<(Condvar, Mutex<(usize, usize)>)>, 
        return_grads: bool,
        _neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>
    )
    {
        // Check the number of visits to this neuron is the same as
        // number of edges to determine if this neuron has 
        // accumulated gradients from all its previous edges.
        if self.attr.get_visit_count(false) == self.get_forward_edges().len()
        {
            input_backward(&mut self.attr, lr, edge_counter, return_grads);
        }
    }

    /// Increment this neuron's sum.
    fn add_to_sum(&mut self, value: f32, is_forward: bool) 
    {
        self.attr.add_to_sum(value, is_forward);
    }

    /// Obtain the current sum of this neuron.
    fn get_sum(&self, is_forward: bool) -> f32 {
        return self.attr.get_sum(is_forward);
    }

    /// Reset the sum of this neuron to zero.
    fn zero_sum(&mut self, is_forward: bool) {
        self.attr.zero_sum(is_forward);
    }

    /// Increment this neuron's visit count.
    fn add_visit_count(&mut self, is_forward: bool) 
    {
        self.attr.add_visit_count(is_forward);
    }

    /// Obtain the current visit count of this neuron.
    fn get_visit_count(&self, is_forward: bool) -> usize 
    {
        return self.attr.get_visit_count(is_forward);
    }

    /// Reset the visit count of this neuron to zero.
    fn zero_visit_count(&mut self, is_forward: bool) 
    {
        self.attr.zero_visit_count(is_forward);
    }

    /// Getter methods to display neuron edges.
    fn get_forward_edges(&self) -> &HashMap<String, ArcEdgeTrait> {
        return &self.attr.forward_edges;
    }

    fn get_backward_edges(&self) -> &HashMap<String, ArcEdgeTrait> {
        return &self.attr.backward_edges;
    }
}