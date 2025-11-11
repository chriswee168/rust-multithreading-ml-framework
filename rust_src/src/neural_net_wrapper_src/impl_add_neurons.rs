
use crate::{neural_net_src::types_aliases::ArcNeuronTrait, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

impl NeuralNetWrapper
{
    // Methods to add neurons.
    pub fn add_input_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.neural_net.add_input_neuron(neuron_id, neuron);
    }

    pub fn add_hidden_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.neural_net.add_hidden_neuron(neuron_id, neuron);
    }

    pub fn add_output_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.neural_net.add_output_neuron(neuron_id, neuron);
    }
}