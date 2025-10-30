use crate::neural_net_src::{neural_net::NeuralNet, types_aliases::ArcNeuronTrait};

impl NeuralNet
{
    // Methods to add neurons.
    pub fn add_input_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.input_neurons.insert(neuron_id, neuron);
    }

    pub fn add_hidden_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.hidden_neurons.insert(neuron_id, neuron);
    }

    pub fn add_output_neuron(&mut self, neuron_id: String, neuron: ArcNeuronTrait)
    {
        self.output_neurons.insert(neuron_id, neuron);
    }
}