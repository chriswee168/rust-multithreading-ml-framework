use std::sync::MutexGuard;

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Remove an edge.
    pub fn remove_edge(&mut self, edge_id: &str)
    {
        // Get edge arc and type.
        let (edge_type, edge) = 
            self.obtain_edge(edge_id);
        let edge: ArcEdgeTrait = edge.unwrap();
        let edge: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();

        let io_neuron: ArcNeuronTrait;
        if edge_type == "input" || edge_type == "output"
        {
            if edge_type == "input"
            {
                // Get the input neuron that uses this edge.
                io_neuron = edge.get_next_neuron().unwrap();
            }
            else
            {
                // Get the output neuron that uses this edge.
                io_neuron = edge.get_prev_neuron().unwrap();
            }
            
            let mut io_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = io_neuron.lock().unwrap();

            if edge_type == "input"
            {
                // Remove edge arc from input neuron.
                io_neuron.remove_backward_edge(edge_id);
                // Remove edge from input edge hashmap.
                self.input_edges.remove(edge_id);
            }
            else
            {
                // Remove edge arc from output neuron.
                io_neuron.remove_forward_edge(edge_id);
                // Remove edge from output edge hashmap.
                self.output_edges.remove(edge_id);
            }
        }
        else if edge_type == "hidden"
        {
            // Get the two neurons that are connected via this edge.
            let neuron0: ArcNeuronTrait = edge.get_prev_neuron().unwrap();
            let neuron1: ArcNeuronTrait = edge.get_next_neuron().unwrap();

            let mut neuron0: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron0.lock().unwrap();
            let mut neuron1: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron1.lock().unwrap();

            // Remove the edge arcs from the two neurons connected.
            neuron0.remove_forward_edge(edge_id);
            neuron1.remove_backward_edge(edge_id);

            // Remove the edge from hidden edge hashmap.
            self.hidden_edges.remove(edge_id);
        }
    }
}