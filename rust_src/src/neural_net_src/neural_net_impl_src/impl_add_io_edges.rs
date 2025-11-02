use std::sync::{Arc, Mutex, MutexGuard};

use crate::neural_net_src::{edge_src::{create_edge::{create_input_edge, create_output_edge}, element_mutexed_vec::ElementMutexedVec, input_edge::InputEdge, output_edge::OutputEdge}, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, rand_id_gen::rand_id_gen, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Add an edge for an input neuron to connect it to an index
    /// of the input array.
    pub fn add_input_edge(
        &mut self, input_neuron_id: String, edge_id_len: usize, 
        input_array_idx: usize, edge_weight_range: f32,
        input_mutexed_vec: Arc<ElementMutexedVec<f32>>
    )
    {
        // Get input neuron arc.
        let input_neuron: ArcNeuronTrait = self.obtain_neuron(input_neuron_id).unwrap();
        // Randomly generate an ID for input edge.
        let random_edge_id: String = rand_id_gen(edge_id_len);

        // Create input edge.
        let input_edge: ArcEdgeTrait = create_input_edge(
            input_array_idx, &input_neuron, input_mutexed_vec, edge_weight_range
        );
        
        // Add edge to beginning of input neuron.
        let mut input_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = input_neuron.lock().unwrap();
        input_neuron.add_backward_edge(random_edge_id.clone(), input_edge.clone());

         // Add the edge to input edge hashmap.
        self.input_edges.insert(random_edge_id, input_edge);
    }

    /// Add an edge for an output neuron to connect it to an index
    /// of the output array.
    pub fn add_output_edge(
        &mut self, output_neuron_id: String, edge_id_len: usize, 
        output_array_idx: usize, edge_weight_range: f32,
        output_mutexed_vec: Arc<ElementMutexedVec<f32>>
    )
    {
        // Get output neuron arc.
        let output_neuron: ArcNeuronTrait = self.obtain_neuron(output_neuron_id).unwrap();
        // Randomly generate an ID for output edge.
        let random_edge_id: String = rand_id_gen(edge_id_len);

        // Create output edge.
        let output_edge: ArcEdgeTrait = create_output_edge(
            &output_neuron, output_array_idx, output_mutexed_vec, edge_weight_range
        );
        
        // Add edge to end of output neuron.
        let mut output_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = output_neuron.lock().unwrap();
        output_neuron.add_forward_edge(random_edge_id.clone(), output_edge.clone());

         // Add the edge to output edge hashmap.
        self.output_edges.insert(random_edge_id, output_edge);
    }
}