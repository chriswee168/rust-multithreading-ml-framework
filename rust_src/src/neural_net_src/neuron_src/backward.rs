use std::{cell::{RefCell, RefMut}, sync::{atomic::AtomicUsize, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::{ArcNeuronTrait, NeuronBuffer}};

/// Calculate gradients of input edge parameters, as well as the final
/// input gradient vector if selected.
pub fn input_backward(
    neuron_attr: &mut NeuronAttr, lr: f32, 
    edge_counter: &Arc<(Condvar, Mutex<(usize, usize)>)>,
    return_grads: bool
)
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

        // Update the output edge count.
        let mut edge_counts_guard: MutexGuard<'_, (usize, usize)> = edge_counter.1.lock().unwrap();
        edge_counts_guard.0 += 1;
                
        // If this is the last output edge being visited, notify the main
        // thread to resume the neural network's forward method.
        if edge_counts_guard.0 == edge_counts_guard.1
        {
            edge_counter.0.notify_one();
        }
    }
}

/// Calculate gradients of backward edges of hidden neurons.
pub fn hidden_backward(neuron_attr: &mut NeuronAttr, lr: f32, neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>)
{
    // Get neuron gradient sum and reset the visit count of this neuron.
    let chained_grad: f32 = neuron_attr.get_sum(false);
    neuron_attr.zero_visit_count(false);
    
    for (_, edge) in &neuron_attr.backward_edges
    {
        let mut edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();

        let prev_neuron: ArcNeuronTrait = edge_guard.get_prev_neuron().unwrap();
        let mut prev_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = prev_neuron.lock().unwrap();

        let prev_neuron_input_val: f32 = prev_neuron_guard.get_sum(true);
        let input_grad: f32 = edge_guard.backward(prev_neuron_input_val, chained_grad, lr);

        let backward_count: usize = prev_neuron_guard.get_visit_count(false);
        if backward_count == 0
        {
            prev_neuron_guard.zero_sum(false);
        }

        prev_neuron_guard.add_to_sum(input_grad, false);
        prev_neuron_guard.add_visit_count(false);

        neuron_buffer.push_back(prev_neuron.clone());
    }
}