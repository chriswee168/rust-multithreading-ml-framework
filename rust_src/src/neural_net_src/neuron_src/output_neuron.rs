use std::{collections::HashMap, sync::RwLockWriteGuard};

use crate::neural_net_src::{neuron_src::{backward::{hidden_backward, output_backward}, core_deps::{NeuronAttr, NeuronTrait}, forward::output_forward}, types_aliases::{ArcEdgeTrait, NeuronBuffer}};

/// Struct to define output neuron attributes and behaviour.
pub struct OutputNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl OutputNeuron
{
    pub fn new(
       neuron_id: String, max_backward_edges: usize, max_forward_edges: usize, 
    ) -> Self
    {
        let neuron_attrs: NeuronAttr = NeuronAttr::new(
            neuron_id, max_backward_edges, max_forward_edges, 
        );

        return Self
        {
            attr: neuron_attrs,
        }
    }
}

impl NeuronTrait for OutputNeuron
{
    /// Add an edge for this neuron to connect to another neuron.
    fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool
    {
        return self.attr.add_forward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from another neuron.
    fn remove_forward_edge(&mut self, edge_id: &str) 
    {
        self.attr.remove_forward_edge(edge_id);
    }
    /// Add an edge for this neuron to connect to a previous neuron.
    fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool
    {
        return self.attr.add_backward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from a previous neuron.
    fn remove_backward_edge(&mut self, edge_id: &str) 
    {
        self.attr.remove_backward_edge(edge_id);
    }
    /// Obtain forward edge max.
    fn get_forward_edge_max(&self) -> usize 
    {
        return self.attr.max_forward_edges;
    }
    /// Obtain backward edge max.
    fn get_backward_edge_max(&self) -> usize 
    {
        return self.attr.max_backward_edges;
    }
    /// Perform forward pass.
    fn forward(&mut self, _neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>) 
    {
        // Check the number of visits to this neuron is the same as
        // number of edges to determine if this neuron has 
        // obtained the full dot product from all its previous edges.
        if self.attr.get_visit_count(true) == self.get_backward_edges().len()
        {
            output_forward(&mut self.attr);
        }
    }
    fn backward(
        &mut self, 
        lr: f32,
        _return_grads: bool,
        neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>
    )
    {
        // Accumulate gradients from output gradient vector first.
        output_backward(&mut self.attr, lr);

        // Backpropagate the gradients through previous neurons.
        hidden_backward(&mut self.attr, lr, neuron_buffer);
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

    fn get_neuron_id(&self) -> String 
    {
        return self.attr.neuron_id.clone();
    }
}