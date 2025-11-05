use std::{collections::HashMap, sync::{Arc, Mutex}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::{core_deps::{NeuronAttr, NeuronTrait}, forward::output_forward}, types_aliases::ArcEdgeTrait};

/// Struct to define output neuron attributes and behaviour.
pub struct OutputNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl OutputNeuron
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

impl NeuronTrait for OutputNeuron
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
    fn forward(&mut self) 
    {
        output_forward(&self.attr);
    }
    fn backward(&mut self) 
    {
        
    }

    /// Increment this neuron's forward sum.
    fn add_forward_sum(&mut self, value: f32) 
    {
        self.attr.add_forward_sum(value);    
    }

    /// Obtain the current forward sum of this neuron.
    fn get_forward_sum(&self) -> f32 {
        return self.attr.get_forward_sum();
    }

    /// Reset the forward sum of this neuron to zero.
    fn zero_forward_sum(&mut self) {
        self.attr.zero_forward_sum();
    }

    /// Increment this neuron's backward sum.
    fn add_backward_sum(&mut self, value: f32) 
    {
        self.attr.add_backward_sum(value);    
    }

    /// Obtain the current backward sum of this neuron.
    fn get_backward_sum(&self) -> f32 {
        return self.attr.get_backward_sum();
    }

    /// Reset the backward sum of this neuron to zero.
    fn zero_backward_sum(&mut self) {
        self.attr.zero_backward_sum();
    }

    /// Getter methods to display neuron edges.
    fn get_forward_edges(&self) -> &HashMap<String, ArcEdgeTrait> {
        return &self.attr.forward_edges;
    }

    fn get_backward_edges(&self) -> &HashMap<String, ArcEdgeTrait> {
        return &self.attr.backward_edges;
    }
}