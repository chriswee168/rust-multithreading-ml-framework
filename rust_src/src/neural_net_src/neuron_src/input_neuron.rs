use crate::neural_net_src::{neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::ArcEdgeTrait};

/// Struct to define input neuron attributes and behaviour.
pub struct InputNeuron
{
    attr: NeuronAttr, // Default neuron attributes.
}

impl InputNeuron
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

impl NeuronTrait for InputNeuron
{
    /// Add an edge for this neuron to connect to another neuron.
    fn add_edge(&mut self, edge: ArcEdgeTrait) 
    {
        self.attr.add_edge(edge);
    }
    /// Remove an edge to disconnect this neuron from another neuron.
    fn remove_edge(&mut self, edge_index: usize) 
    {
        self.attr.remove_edge(edge_index);
    }
    fn forward(&mut self) 
    {
        
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