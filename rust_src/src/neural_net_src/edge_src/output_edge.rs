use std::sync::Arc;

use crate::neural_net_src::{edge_src::{core_deps::{EdgeAttr, EdgeTrait}, element_mutexed_vec::ElementMutexedVec}, types_aliases::ArcNeuronTrait};

/// Connects an output neuron with an index of output array.
pub struct OutputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    prev_neuron: ArcNeuronTrait,
    output_array_index: usize,

    // Used during forward propagation to obtain output array.
    output_mutexed_vec: Arc<ElementMutexedVec<f32>>,
}

// Implement constructor.
impl OutputEdge
{
    pub fn new(
        prev_neuron: ArcNeuronTrait, 
        output_array_index: usize, 
        output_mutexed_vec: Arc<ElementMutexedVec<f32>>,
        weight_range: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(weight_range);
        return Self
        {
            attr: edge_attr,
            prev_neuron,
            output_array_index,
            output_mutexed_vec
        };        
    }
}

impl EdgeTrait for OutputEdge
{
    // Forward propagation.
    fn forward(&self, input_val: f32) -> f32 {
        // Perform y = w * x + b
        if input_val >= 0.0
        {
            return input_val * self.attr.pos_weight + self.attr.bias
        }
        else
        {
            return input_val * self.attr.neg_weight + self.attr.bias
        }
    }

    // Backpropagation through chain rule.
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32 {
        // Derivatives
        // y = w * x + b
        // d_y/d_x = w
        // d_y/d_w = x
        // d_y/d_b = 1

        // Update bias parameter.
        self.attr.bias -= lr * gradient_val;

        // Update weight parameter and calculate the gradient respect to input.
        let input_gradient: f32;
        if input_val >= 0.0
        {
            self.attr.pos_weight -= lr * gradient_val * input_val;
            input_gradient = gradient_val * self.attr.pos_weight;
        }
        else
        {
            self.attr.neg_weight -= lr * gradient_val * input_val;
            input_gradient = gradient_val * self.attr.neg_weight;
        }

        return input_gradient;
    }

    // Get previous output neuron this edge is connected to.
    fn get_prev_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.prev_neuron));
    }

    // Get index of output array.
    fn get_next_id(&self) -> Option<usize> {
        return Some(self.output_array_index);
    }
}