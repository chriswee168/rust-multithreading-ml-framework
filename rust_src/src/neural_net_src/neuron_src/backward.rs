use std::{cell::{RefCell, RefMut}, sync::{atomic::AtomicUsize, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::{ArcNeuronTrait, NeuronBuffer}};

/// Calculate gradients of input edge parameters, as well as the final
/// input gradient vector if selected.
pub fn input_backward(neuron_attr: &mut NeuronAttr, lr: f32, return_grads: bool)
{   
    // Calculate the gradients for parameters of each edge to return to the input 
    // gradient vector.
    let chained_grad: f32 = neuron_attr.get_sum(false);
    for (_, edge) in &neuron_attr.backward_edges
    {
        // Get edge guard for exclusive access.
        let mut edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
        
        // Get read guard of input vector.
        let input_rwlock_vec: Arc<RwLock<Vec<f32>>> = edge_guard.get_rwlock_vec().unwrap();
        let input_read_guard: RwLockReadGuard<'_, Vec<f32>> = input_rwlock_vec.read().unwrap();

        // Calculate the gradients for this edge's parameters and perform gradient descent.
        let vector_index: usize = edge_guard.get_prev_id().unwrap();
        let input_value: f32 = input_read_guard[vector_index];
        let input_grad: f32 = edge_guard.backward(input_value, chained_grad, lr);

        // Update the input gradient array if selected.
        if return_grads
        {
            // Get read guard of input gradient vector.
            let grad_rwlock_vec: Arc<RwLock<Vec<f32>>> = edge_guard.get_grad_rwlock_vec().unwrap();
            let mut grad_read_guard: RwLockWriteGuard<'_, Vec<f32>> = grad_rwlock_vec.write().unwrap();

            // Accumulate the gradient at the specific index.
            grad_read_guard[vector_index] += input_grad;
        }
    }

    // Zero the input neuron gradient sum.
    neuron_attr.zero_sum(false);
}