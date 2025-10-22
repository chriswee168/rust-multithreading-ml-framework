use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::core_deps::{EdgeAttr, EdgeTrait}, neuron_src::core_deps::NeuronTrait};

/// Connects an index of the input array with an input neuron.
pub struct OutputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    prev_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>,
    output_array_index: usize,
}

// Implement constructor.
impl OutputEdge
{
    pub fn new(
        prev_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>, 
        output_array_index: usize, 
        weight_range: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(weight_range);
        return Self
        {
            attr: edge_attr,
            prev_neuron,
            output_array_index
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
}