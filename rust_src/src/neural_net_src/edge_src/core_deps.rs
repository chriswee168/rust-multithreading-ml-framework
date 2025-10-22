use std::sync::{Arc, Mutex};

use rand::Rng;

use crate::neural_net_src::neuron_src::core_deps::NeuronTrait;

/// Struct that defines default attributes for edges.
pub struct EdgeAttr
{
    // Parameters to use during forward propagation.
    // y = w * x + b
    // Weight values to use for negative and positive input.
    neg_weight: f32,
    pos_weight: f32,
    bias: f32, // Bias value to shift the product.

    // Gradients to obtain from back propagation to optimize parameters.
    neg_weight_grad: f32,
    pos_weight_grad: f32,
    bias_grad: f32,
}

// Implement constructor method.
impl EdgeAttr
{
    pub fn new(weight_range: f32) -> Self
    {
        // Used for random weight initialization within specified range.
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        return Self
        {
            neg_weight: rand_gen.gen_range(-weight_range..=weight_range),
            pos_weight: rand_gen.gen_range(-weight_range..=weight_range),
            bias: 0.0,
            neg_weight_grad: 0.0,
            pos_weight_grad: 0.0,
            bias_grad: 0.0
        }
    }
}