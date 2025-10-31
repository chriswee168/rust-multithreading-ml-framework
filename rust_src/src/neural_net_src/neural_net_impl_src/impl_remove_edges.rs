use std::sync::MutexGuard;

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neural_net::NeuralNet, neuron_src::{core_deps::NeuronTrait, input_neuron}, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Remove a hidden edge.
    pub fn remove_hidden_edge(&mut self, edge_id: String)
    {
        let hidden_edge: ArcEdgeTrait = self.obtain_edge(edge_id.clone()).unwrap();
        let hidden_edge: MutexGuard<'_, Box<dyn EdgeTrait>> = hidden_edge.lock().unwrap();
        
        // Get the two neurons that are connected via this edge.
        let neuron0: ArcNeuronTrait = hidden_edge.get_prev_neuron().unwrap();
        let neuron1: ArcNeuronTrait = hidden_edge.get_next_neuron().unwrap();

        let mut neuron0: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron0.lock().unwrap();
        let mut neuron1: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron1.lock().unwrap();

        // Remove the edge arcs from the two neurons connected.
        neuron0.remove_forward_edge(edge_id.clone());
        neuron1.remove_backward_edge(edge_id.clone());

        // Remove the edge from hidden edge hashmap.
        self.hidden_edges.remove(&edge_id);
    }

    /// Remove an input edge.
    pub fn remove_input_edge(&mut self, edge_id: String)
    {
        let input_edge: ArcEdgeTrait = self.obtain_edge(edge_id.clone()).unwrap();
        let input_edge: MutexGuard<'_, Box<dyn EdgeTrait>> = input_edge.lock().unwrap();

        // Get the input neuron that uses this edge.
        let input_neuron: ArcNeuronTrait = input_edge.get_next_neuron().unwrap();
        let mut input_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = input_neuron.lock().unwrap();

        // Remove edge arc from input neuron.
        input_neuron.remove_backward_edge(edge_id.clone());

        // Remove edge from input edge hashmap.
        self.input_edges.remove(&edge_id);
    }
}