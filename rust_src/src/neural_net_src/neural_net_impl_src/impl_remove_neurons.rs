use std::sync::MutexGuard;

use crate::neural_net_src::{neuron_src::core_deps::NeuronTrait, neural_net::NeuralNet, types_aliases::{ArcNeuronTrait}};

impl NeuralNet
{
    /// Remove a neuron if it has no edges.
    pub fn remove_empty_neuron(&mut self, neuron_id: &str)
    {
        let (neuron_type, neuron) = 
            self.obtain_neuron(neuron_id);
        
        let neuron: ArcNeuronTrait = neuron.unwrap();
        let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
        let is_empty: bool = neuron_guard.get_forward_edges().is_empty() && 
            neuron_guard.get_backward_edges().is_empty();
        drop(neuron_guard);

        if is_empty
        {
            if neuron_type == "input"
            {
                self.input_neurons.remove(neuron_id);
            }
            else if neuron_type == "output"
            {
                self.output_neurons.remove(neuron_id);
            }
            else if neuron_type == "hidden"
            {
                self.hidden_neurons.remove(neuron_id);
            }
        }
    }
}