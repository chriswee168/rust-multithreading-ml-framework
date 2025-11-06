use std::sync::{Arc, RwLock};

use crate::neural_net_src::{edge_src::{core_deps::{EdgeAttr, EdgeTrait}}, types_aliases::ArcNeuronTrait};

/// Connects an output neuron with an index of output array.
pub struct OutputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    prev_neuron: ArcNeuronTrait,
    output_array_index: usize,

    // Arc pointer which stores the output array to return.
    output_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    // Arc pointer for parallel read access of output array gradients during backpropagation.
    grad_rwlock_vec: Arc<RwLock<Vec<f32>>>,
}

// Implement constructor.
impl OutputEdge
{
    pub fn new(
        prev_neuron: ArcNeuronTrait, 
        output_array_index: usize, 
        output_rwlock_vec: Arc<RwLock<Vec<f32>>>,
        grad_rwlock_vec: Arc<RwLock<Vec<f32>>>,
        neg_weight: f32,
        pos_weight: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(neg_weight, pos_weight);
        return Self
        {
            attr: edge_attr,
            prev_neuron,
            output_array_index,
            output_rwlock_vec,
            grad_rwlock_vec
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

    // Obtain output rwlock vector for obtaining output values during forward pass.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.output_rwlock_vec));
    }
}