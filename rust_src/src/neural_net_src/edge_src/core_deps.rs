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
    pub bias: f32, // Bias value to shift the product.
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
            bias: 0.0,
        }
    }
}

/// Contains trait methods for input, hidden and output edges.
pub trait EdgeTrait
{
    fn forward(&self, input_val: f32) -> f32;
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32;

    // Methods to obtain the previous/next neuron/index, typing depends on
    // edge struct.
    fn get_prev_id(&self) -> Option<usize> { None }
    fn get_prev_neuron(&self) -> Option<ArcNeuronTrait> { None }
    fn get_next_id(&self) -> Option<usize> { None }
    fn get_next_neuron(&self) -> Option<ArcNeuronTrait> { None }

    // Obtain the rwlock vector for input and output edges.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> { None }
}