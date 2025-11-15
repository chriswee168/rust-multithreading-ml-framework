use std::{cell::{RefCell, RefMut}, sync::{atomic::AtomicUsize, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::{ArcNeuronTrait, NeuronBuffer}};

/// Forward propagation method explicitly for input neurons to work on values
/// directly from the input array via input edges.
pub fn input_forward(neuron_attr: &mut NeuronAttr)
{
    // Zero the input neuron sum.
    neuron_attr.zero_sum(true);

    // Get the total sum or dot product of input values multiplied.
    let mut total_sum: f32 = 0.0;
    for (_, edge) in &neuron_attr.backward_edges
    {
        let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
        let rwlock_vec: Arc<RwLock<Vec<f32>>> = edge_guard.get_rwlock_vec().unwrap();
        let read_guard: RwLockReadGuard<'_, Vec<f32>> = rwlock_vec.read().unwrap();
        let vector_index: usize = edge_guard.get_prev_id().unwrap();
        let input_value: f32 = read_guard[vector_index];
        let output_value: f32 = edge_guard.forward(input_value);

        total_sum += output_value;
    }

    // Set the input neuron value for propagation to hidden neurons.
    neuron_attr.add_to_sum(total_sum, true);
}

/// Function for input/hidden neurons to forward propagate values though each
/// edge.
pub fn hidden_forward(neuron_attr: &mut NeuronAttr, neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>)
{
    // Get neuron sum and reset the visit count of this neuron.
    let neuron_sum: f32 = neuron_attr.get_sum(true);
    neuron_attr.zero_visit_count(true);

    for (_, edge) in &neuron_attr.forward_edges
    {
        // Reference to the next neuron this edge connects to.
        let next_neuron: ArcNeuronTrait;

        // Value to increment the next neuron's received sum.
        let edge_output: f32;

        {
            // Get exclusive access to edge.
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            edge_output = edge_guard.forward(neuron_sum);

            // Get the next neuron.
            next_neuron = edge_guard.get_next_neuron().unwrap();
        }

        {
            // Get exclusive access to the next neuron.
            let mut next_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = next_neuron.lock().unwrap();
            
            // Get forward visit count of next neuron.
            let visit_count: usize = next_neuron_guard.get_visit_count(true);

            // Reset the neuron sum of the next neuron if this is the first
            // neuron to visit it.
            if visit_count == 0
            {
                next_neuron_guard.zero_sum(true);
            }

            // Increment the edge output value to the received sum of next neuron.
            next_neuron_guard.add_to_sum(edge_output, true);
            next_neuron_guard.add_visit_count(true);
        }

        // Append the next neuron to the thread's neuron buffer.
        neuron_buffer.push_back(next_neuron.clone());
    }
}

/// Function for output neurons to forward propagate values though each
/// edge. Output neurons has output edges containing indexes for output
/// array.
pub fn output_forward(neuron_attr: &mut NeuronAttr, edge_counter: &Arc<(Condvar, Mutex<(usize, usize)>)>)
{
    // Get neuron sum and reset the visit count of this neuron.
    let neuron_sum: f32 = neuron_attr.get_sum(true);
    neuron_attr.zero_visit_count(true);
    
    for (_, edge) in &neuron_attr.forward_edges
    {
        // The output index of the output array this edge "connects" to.
        let output_index: usize;
        // Value to write to the index of the output array.
        let edge_output: f32;
        // Obtains the output values to send to the python frontend as an
        // array.
        let output_rwlock_vec: Arc<RwLock<Vec<f32>>>;

        {
            // Get exclusive access to edge.
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            edge_output = edge_guard.forward(neuron_sum);

            // Get the output index of the output array this edge "connects" to.
            output_index = edge_guard.get_next_id().unwrap();
            output_rwlock_vec = edge_guard.get_rwlock_vec().unwrap();
        }

        {
            // Get exclusive access to specific index of output array and write
            // edge output.
            let mut output_array: RwLockWriteGuard<'_, Vec<f32>> = output_rwlock_vec.write().unwrap();
            output_array[output_index] += edge_output;

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
}