use std::sync::MutexGuard;

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::{NeuronAttr, NeuronTrait}, types_aliases::ArcNeuronTrait};

/// Function for input/hidden neurons to forward propagate values though each
/// edge.
pub fn hidden_forward(neuron_attr: &NeuronAttr)
{
    for edge in &neuron_attr.forward_edges
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