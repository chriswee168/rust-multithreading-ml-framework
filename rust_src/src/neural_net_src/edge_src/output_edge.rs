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
        return self.forward_def(
            input_val, 
            self.attr.pos_weight, 
            self.attr.neg_weight, 
            self.attr.neg_bias,
            self.attr.pos_bias
        );
    }

    // Backpropagation through chain rule.
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32 {
        let (input_grad, new_neg_bias, new_pos_bias, new_pos_weight, new_neg_weight) = 
            self.backward_def(&self.attr, input_val, gradient_val, lr);

        // Update the edge parameters.
        self.attr.neg_bias = new_neg_bias;
        self.attr.pos_bias = new_pos_bias;
        self.attr.pos_weight = new_pos_weight;
        self.attr.neg_weight = new_neg_weight;

        return input_grad;
    }

    // Get previous output neuron this edge is connected to.
    fn get_prev_neuron(&self) -> Option<ArcNeuronTrait> {
        return Some(Arc::clone(&self.prev_neuron));
    }

    // Get index of output array.
    fn get_next_idx(&self) -> Option<usize> {
        return Some(self.output_array_index);
    }

    // Obtain output rwlock vector to update the output array during forward pass.
    fn get_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.output_rwlock_vec));
    }

    // Obtain output gradient array during backward pass.
    fn get_grad_rwlock_vec(&self) -> Option<Arc<RwLock<Vec<f32>>>> {
        return Some(Arc::clone(&self.grad_rwlock_vec));
    }

    // Get parameters.
    fn get_params(&self) -> (f32, f32, f32, f32) {
        return (self.attr.pos_weight, self.attr.neg_weight, self.attr.neg_bias, self.attr.pos_bias);
    }

    // Set the negative and positive weights.
    fn set_params(&mut self, pos_weight: f32, neg_weight: f32) 
    {
        self.attr.pos_weight = pos_weight;
        self.attr.neg_weight = neg_weight;
    }
}