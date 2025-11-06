use std::sync::{Arc, RwLock};

use crate::neural_net_src::{edge_src::{core_deps::{EdgeAttr, EdgeTrait}}, types_aliases::ArcNeuronTrait};

/// Connects an index of the input array with an input neuron.
pub struct InputEdge
{
    attr: EdgeAttr, // Contains the essential attributes of an edge.
    input_array_index: usize,

    // Arc pointer to the input array, allows read access between threads in parallel.
    input_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    // Arc pointer to store the input array gradients during backpropagation.
    grad_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    next_neuron: ArcNeuronTrait
}

// Implement constructor.
impl InputEdge
{
    pub fn new(
        input_array_index: usize, 
        next_neuron: ArcNeuronTrait, 
        input_rwlock_vec: Arc<RwLock<Vec<f32>>>,
        grad_rwlock_vec: Arc<RwLock<Vec<f32>>>,
        neg_weight: f32,
        pos_weight: f32
    ) -> Self
    {
        let edge_attr: EdgeAttr = EdgeAttr::new(neg_weight, pos_weight);
        return Self
        {
            attr: edge_attr,
            input_array_index,
            input_rwlock_vec,
            grad_rwlock_vec,
            next_neuron
        };        
    }
}

impl EdgeTrait for InputEdge
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

    // Get input array index.
    fn get_prev_id(&self) -> Option<usize> {
        return Some(self.input_array_index);
    }

    // Get the next input neuron this edge connections
    fn get_next_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.next_neuron));
    }

    // Obtain input rwlock vector for obtaining input gradients
    // during backpropagation.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.input_rwlock_vec));
    }
}