use std::sync::{Arc, Mutex, MutexGuard};

use crate::neural_net_src::{edge_src::{create_edge::create_hidden_edge, hidden_edge::HiddenEdge}, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, rand_id_gen::rand_id_gen, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Join two neurons with a hidden edge.
    pub fn join_neurons(
        &mut self, neuron0_id: &str, neuron1_id: &str, 
        edge_id: String, neg_weight: f32, pos_weight: f32
    )
    {
        // Get the neuron arcs.
        let (_, neuron0) = self.obtain_neuron(neuron0_id);
        let (_, neuron1) = self.obtain_neuron(neuron1_id);
        let neuron0: ArcNeuronTrait = neuron0.unwrap();
        let neuron1: ArcNeuronTrait = neuron1.unwrap();
        
        // Create hidden edge.
        let hidden_edge: ArcEdgeTrait = create_hidden_edge(
            &neuron0, &neuron1, neg_weight, pos_weight
        );

        // Acquire mutexes for mutability.
        let mut neuron0: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron0.lock().unwrap();
        let mut neuron1: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron1.lock().unwrap();

        if neuron0.get_forward_edges().len() < neuron0.get_forward_edge_max() &&
           neuron1.get_backward_edges().len() < neuron1.get_backward_edge_max()
        {
            // Connect the two neurons with the same hidden edge.
            let edge1_added: bool = neuron0.add_forward_edge(edge_id.clone(), hidden_edge.clone());
            let edge2_added: bool = neuron1.add_backward_edge(edge_id.clone(), hidden_edge.clone());

            if edge1_added && edge2_added
            {
                // Add the edge to hidden edge hashmap.
                self.hidden_edges.insert(edge_id, hidden_edge);
            }
        }
    }
}