use std::{cell::{RefCell, RefMut}, sync::{Arc, MutexGuard}};

use crate::neural_net_src::{edge_src::{core_deps::EdgeTrait, element_mutexed_vec::ElementMutexedVec}, neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::ArcNeuronTrait};

/// Function for input/hidden neurons to forward propagate values though each
/// edge.
pub fn hidden_forward(neuron_attr: &NeuronAttr)
{
    for (_, edge) in &neuron_attr.forward_edges
    {
        // Reference to the next neuron this edge connects to.
        let next_neuron: ArcNeuronTrait;

        // Value to increment the next neuron's received sum.
        let edge_output: f32;

        {
            // Get exclusive access to edge.
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            edge_output = edge_guard.forward(neuron_attr.get_sum());

            // Get the next neuron.
            next_neuron = edge_guard.get_next_neuron().unwrap();
        }

        {
            // Get exclusive access to the next neuron.
            let mut next_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = next_neuron.lock().unwrap();
                
            // Increment the edge output value to the received sum of next neuron.
            next_neuron_guard.increment_sum(edge_output);
        }
    }
}

/// Function for output neurons to forward propagate values though each
/// edge. Output neurons has output edges containing indexes for output
/// array.
pub fn output_forward(neuron_attr: &NeuronAttr)
{
    for (_, edge) in &neuron_attr.forward_edges
    {
        // The output index of the output array this edge "connects" to.
        let output_index: usize;
        // Value to write to the index of the output array.
        let edge_output: f32;
        // Obtains the output values to send to the python frontend as an
        // array.
        let output_mutexed_vec: Arc<ElementMutexedVec<f32>>;

        {
            // Get exclusive access to edge.
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            edge_output = edge_guard.forward(neuron_attr.get_sum());

            // Get the output index of the output array this edge "connects" to.
            output_index = edge_guard.get_next_id().unwrap();
            output_mutexed_vec = edge_guard.get_element_mutexed_vec().unwrap();
        }

        {
            // Get exclusive access to specific index of output array and write
            // edge output.
            let _guard: MutexGuard<'_, ()> = output_mutexed_vec.locks[output_index].lock().unwrap();
            let mut output_array: RefMut<'_, Vec<f32>> = output_mutexed_vec.data.borrow_mut();
            output_array[output_index] += edge_output;
        }
    }
}