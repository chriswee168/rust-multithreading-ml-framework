use crate::neural_net_src::{neuron_src::{core_deps::{NeuronAttr, NeuronTrait}, forward::hidden_forward}, types_aliases::ArcEdgeTrait};

/// Struct to define hidden neuron attributes and behaviour.
pub struct HiddenNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl HiddenNeuron
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

impl NeuronTrait for HiddenNeuron
{
    /// Add an edge for this neuron to connect to another neuron.
    fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) 
    {
        self.attr.add_forward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from another neuron.
    fn remove_forward_edge(&mut self, edge_id: String) 
    {
        self.attr.remove_forward_edge(edge_id);
    }
    /// Add an edge for this neuron to connect to a previous neuron.
    fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) 
    {
        self.attr.add_backward_edge(edge_id, edge);
    }
    /// Remove an edge to disconnect this neuron from a previous neuron.
    fn remove_backward_edge(&mut self, edge_id: String) 
    {
        self.attr.remove_backward_edge(edge_id);
    }
    /// Perform forward pass.
    fn forward(&mut self) 
    {
        hidden_forward(&self.attr);
    }
    fn backward(&mut self) 
    {
        
    }

    /// Increment this neuron's received sum.
    fn increment_sum(&mut self, value: f32) 
    {
        self.attr.increment_sum(value);    
    }

    /// Obtain the current received sum of this neuron.
    fn get_sum(&self) -> f32 {
        return self.attr.get_sum();
    }

    /// Reset the received sum of this neuron to zero.
    fn zero_sum(&mut self) {
        self.attr.zero_sum();
    }
}