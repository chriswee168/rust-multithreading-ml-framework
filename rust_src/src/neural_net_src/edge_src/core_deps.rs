use std::sync::{Arc, RwLock};

use crate::neural_net_src::{types_aliases::ArcNeuronTrait};

/// Struct that defines default attributes for edges.
pub struct EdgeAttr
{
    // Parameters to use during forward propagation.
    // y = w * x + b
    // Weight values to use for negative and positive input.
    pub neg_weight: f32,
    pub pos_weight: f32,
}

// Implement constructor method.
impl EdgeAttr
{
    pub fn new(neg_weight: f32, pos_weight: f32) -> Self
    {
        return Self
        {
            neg_weight,
            pos_weight,
        }
    }
}

/// Contains trait methods for input, hidden and output edges.
pub trait EdgeTrait: Send
{
    fn forward(&self, input_val: f32) -> f32;
    // Default implementation.
    fn forward_def(&self, input_val: f32, pos_weight: f32, neg_weight: f32) -> f32
    {
        // Perform y = w * x
        if input_val >= 0.0
        {
            return input_val * pos_weight;
        }
        else
        {
            return input_val * neg_weight;
        }
    }
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32;
    // Default implementation.
    fn backward_def(
        &self, attr: &EdgeAttr,
        input_val: f32, gradient_val: f32, lr: f32
    ) -> (f32, f32, f32) {
        // Derivatives
        // y = w * x
        // d_y/d_x = w
        // d_y/d_w = x

        // Update weight and bias parameters and calculate the gradient respect to input.
        let input_gradient: f32;
        let mut new_pos_weight: f32 = attr.pos_weight;
        let mut new_neg_weight: f32 = attr.neg_weight;
        if input_val >= 0.0
        {
            new_pos_weight -= lr * gradient_val * input_val;
            input_gradient = gradient_val * attr.pos_weight;
        }
        else
        {
            new_neg_weight -= lr * gradient_val * input_val;
            input_gradient = gradient_val * attr.neg_weight;
        }

        return (input_gradient, new_pos_weight, new_neg_weight);
    }

    // Methods to obtain the previous/next neuron/index, typing depends on
    // edge struct.
    fn get_prev_idx(&self) -> Option<usize> { None }
    fn get_prev_neuron(&self) -> Option<ArcNeuronTrait> { None }
    fn get_next_idx(&self) -> Option<usize> { None }
    fn get_next_neuron(&self) -> Option<ArcNeuronTrait> { None }

    // Obtain the rwlock vector for input and output edges.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> { None }
    // Obtain the gradient rwlock vector for input and output edges.
    fn get_grad_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> { None }

    // Obtain the parameters of this edge.
    fn get_params(&self) -> (f32, f32);

    // Set the negative and positive weights.
    fn set_params(&mut self, pos_weight: f32, neg_weight: f32);
}