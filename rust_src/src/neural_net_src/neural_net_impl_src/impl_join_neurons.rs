use std::sync::{Arc, Mutex, MutexGuard};

use crate::neural_net_src::{edge_src::{create_edge::create_hidden_edge, hidden_edge::HiddenEdge}, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, rand_id_gen::rand_id_gen, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Join two neurons with a hidden edge.
    pub fn join_neurons(
        &mut self, neuron0_id: &str, neuron1_id: &str, 
        edge_id_len: usize, edge_weight_range: f32
    )
    {
        // Get the neuron arcs.
        let neuron0: ArcNeuronTrait = self.obtain_neuron(neuron0_id).unwrap();
        let neuron1: ArcNeuronTrait = self.obtain_neuron(neuron1_id).unwrap();

        // Randomly generate an ID for hidden edge.
        let random_edge_id: String = rand_id_gen(edge_id_len);

        // Create hidden edge.
        let hidden_edge: ArcEdgeTrait = create_hidden_edge(
            &neuron0, &neuron1, edge_weight_range
        );

        // Acquire mutexes for mutability.
        let mut neuron0: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron0.lock().unwrap();
        let mut neuron1: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron1.lock().unwrap();

        // Connect the two neurons with the same hidden edge.
        neuron0.add_forward_edge(random_edge_id.clone(), hidden_edge.clone());
        neuron1.add_backward_edge(random_edge_id.clone(), hidden_edge.clone());

        // Add the edge to hidden edge hashmap.
        self.hidden_edges.insert(random_edge_id, hidden_edge);
    }
}