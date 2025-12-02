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
        return self.forward_def(
            input_val, 
            self.attr.pos_weight, 
            self.attr.neg_weight, 
            self.attr.bias
        );
    }

    // Backpropagation through chain rule.
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32 {
        let (input_grad, new_bias, new_pos_weight, new_neg_weight) = 
            self.backward_def(&self.attr, input_val, gradient_val, lr);

        // Update the edge parameters.
        self.attr.bias = new_bias;
        self.attr.pos_weight = new_pos_weight;
        self.attr.neg_weight = new_neg_weight;

        return input_grad;
    }

    // Get input array index.
    fn get_prev_idx(&self) -> Option<usize> {
        return Some(self.input_array_index);
    }

    // Get the next input neuron this edge connections
    fn get_next_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.next_neuron));
    }

    // Obtain input rwlock vector for obtaining input array values
    // during forward propagation.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.input_rwlock_vec));
    }

    // Obtain input rwlock vector to store obtaining input gradients
    // during backpropagation.
    fn get_grad_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.grad_rwlock_vec));
    }

    // Get parameters.
    fn get_params(&self) -> (f32, f32, f32) {
        return (self.attr.pos_weight, self.attr.neg_weight, self.attr.bias);
    }

    // Set the negative and positive weights.
    fn set_params(&mut self, pos_weight: f32, neg_weight: f32) 
    {
        self.attr.pos_weight = pos_weight;
        self.attr.neg_weight = neg_weight;
    }
}