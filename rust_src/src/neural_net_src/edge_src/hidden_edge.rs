use std::sync::Arc;

use crate::neural_net_src::{edge_src::core_deps::{EdgeAttr, EdgeTrait}, types_aliases::ArcNeuronTrait};

/// Connects any two neurons (input->hidden, hidden->hidden, hidden->output).
pub struct HiddenEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    prev_neuron: ArcNeuronTrait,
    next_neuron: ArcNeuronTrait,
}

// Implement constructor.
impl HiddenEdge
{
    pub fn new(
        prev_neuron: ArcNeuronTrait, 
        next_neuron: ArcNeuronTrait, 
        neg_weight: f32,
        pos_weight: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(neg_weight, pos_weight);
        return Self
        {
            attr: edge_attr,
            prev_neuron,
            next_neuron
        };        
    }
}

impl EdgeTrait for HiddenEdge
{
    // Forward propagation.
    fn forward(&self, input_val: f32) -> f32 {
        return self.forward_def(
            input_val, 
            self.attr.pos_weight, 
            self.attr.neg_weight, 
        );
    }

    // Backpropagation through chain rule.
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32 {
        let (input_grad, new_pos_weight, new_neg_weight) = 
            self.backward_def(&self.attr, input_val, gradient_val, lr);

        // Update the edge parameters.
        self.attr.pos_weight = new_pos_weight;
        self.attr.neg_weight = new_neg_weight;

        return input_grad;
    }

    // Get previous neuron this edge connects.
    fn get_prev_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.prev_neuron));
    }

    // Get next neuron this edge connects.
    fn get_next_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.next_neuron));
    }

    // Get parameters.
    fn get_params(&self) -> (f32, f32) {
        return (self.attr.pos_weight, self.attr.neg_weight);
    }

    // Set the negative and positive weights.
    fn set_params(&mut self, pos_weight: f32, neg_weight: f32) 
    {
        self.attr.pos_weight = pos_weight;
        self.attr.neg_weight = neg_weight;
    }
}