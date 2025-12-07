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

    /// Obtain the neuron arc from either input, hidden or output neuron hashmap.
    pub fn obtain_neuron(&self, neuron_id: &str) -> (&str, Option<ArcNeuronTrait>)
    {
        if self.input_neurons.get(neuron_id).is_some()
        {
            return ("input", Some(self.input_neurons.get(neuron_id).unwrap().clone()));
        }
        else if self.hidden_neurons.get(neuron_id).is_some() 
        {
            return ("hidden", Some(self.hidden_neurons.get(neuron_id).unwrap().clone()));
        }
        else if self.output_neurons.get(neuron_id).is_some()
        {
            return ("output", Some(self.output_neurons.get(neuron_id).unwrap().clone()));
        }
        else
        {
            ("none", None)
        }
    }
}