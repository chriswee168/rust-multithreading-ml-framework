use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use crate::neural_net_src::{edge_src::{create_edge::{create_input_edge, create_output_edge}, input_edge::InputEdge, output_edge::OutputEdge}, neural_net::NeuralNet, neuron_src::{core_deps::NeuronTrait, output_neuron}, rand_id_gen::rand_id_gen, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

impl NeuralNet
{
    /// Add an edge for an input neuron to connect it to an index
    /// of the input array.
    pub fn add_input_edge(
        &mut self, input_neuron_id: &str, edge_id: String, 
        input_array_idx: usize, neg_weight: f32, pos_weight: f32,
        input_rwlock_vec: Arc<RwLock<Vec<f32>>>, 
        grad_rwlock_vec: Arc<RwLock<Vec<f32>>>
    )
    {
        // Get input neuron arc.
        let (_, input_neuron) = self.obtain_neuron(input_neuron_id);
        let input_neuron: ArcNeuronTrait = input_neuron.unwrap();

        // Create input edge.
        let input_edge: ArcEdgeTrait = create_input_edge(
            input_array_idx, &input_neuron, 
            input_rwlock_vec, 
            grad_rwlock_vec,
            neg_weight, pos_weight
        );
        
        // Add edge to beginning of input neuron.
        let mut input_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = input_neuron.lock().unwrap();
        input_neuron.add_backward_edge(edge_id.clone(), input_edge.clone());

        if !self.input_edges.contains_key(&edge_id)
        {
            // Add the edge to input edge hashmap.
            self.input_edges.insert(edge_id, input_edge);
        }
    }

    /// Add an edge for an output neuron to connect it to an index
    /// of the output array.
    pub fn add_output_edge(
        &mut self, output_neuron_id: &str, edge_id: String, 
        output_array_idx: usize, neg_weight: f32, pos_weight: f32,
        output_rwlock_vec: Arc<RwLock<Vec<f32>>>,
        grad_rwlock_vec: Arc<RwLock<Vec<f32>>>
    )
    {
        // Get output neuron arc.
        let (_, output_neuron) = self.obtain_neuron(output_neuron_id);
        let output_neuron: ArcNeuronTrait = output_neuron.unwrap();

        // Create output edge.
        let output_edge: ArcEdgeTrait = create_output_edge(
            &output_neuron, output_array_idx, 
            output_rwlock_vec,
            grad_rwlock_vec, 
            neg_weight, pos_weight
        );
        
        // Add edge to end of output neuron.
        let mut output_neuron: MutexGuard<'_, Box<dyn NeuronTrait>> = output_neuron.lock().unwrap();
        output_neuron.add_forward_edge(edge_id.clone(), output_edge.clone());

        if !self.output_edges.contains_key(&edge_id)
        {
            // Add the edge to output edge hashmap.
            self.output_edges.insert(edge_id, output_edge);
        }
    }
}