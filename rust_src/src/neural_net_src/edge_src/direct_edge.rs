use std::sync::{Arc, Mutex};

use crate::neural_net_src::neuron_src::neuron::NeuronTrait;

/// Struct that defines directed edges responsible for connecting
/// neurons.
pub struct DirectEdge
{
    // References to neurons connected to each other.
    prev_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>,
    next_neuron: Arc<Mutex<Box<dyn NeuronTrait>>>,

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