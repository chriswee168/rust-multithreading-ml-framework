use std::sync::Arc;

use crate::neural_net_src::{neural_net::NeuralNet, types_aliases::ArcNeuronTrait};

impl NeuralNet
{
    // Join two neurons with a hidden edge.
    pub fn join_neurons(
        &mut self, neuron0_id: String, neuron1_id: String,
    )
    {
        // Get the neuron arcs.
        let neuron0: ArcNeuronTrait = self.obtain_neuron(neuron0_id).unwrap();
        let neuron1: ArcNeuronTrait = self.obtain_neuron(neuron1_id).unwrap();


    }

    // Obtain the neuron arc from either input, hidden or output neuron hashmap.
    fn obtain_neuron(&self, neuron_id: String) -> Option<ArcNeuronTrait>
    {
        if self.input_neurons.get(&neuron_id).is_some()
        {
            return Some(Arc::clone(self.input_neurons.get(&neuron_id).unwrap()));
        }
        else if self.hidden_neurons.get(&neuron_id).is_some() 
        {
            return Some(Arc::clone(self.hidden_neurons.get(&neuron_id).unwrap()));
        }
        else if self.output_neurons.get(&neuron_id).is_some()
        {
            return Some(Arc::clone(self.output_neurons.get(&neuron_id).unwrap()));
        }
        else
        {
            None
        }
    }
}