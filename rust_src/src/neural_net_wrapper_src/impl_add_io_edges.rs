use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

impl NeuralNetWrapper
{
    /// Add an edge for an input neuron to connect it to an index
    /// of the input array.
    pub fn add_input_edge(
        &mut self, input_neuron_id: &str, edge_id: String, 
        input_array_idx: usize, neg_weight: f32, pos_weight: f32,
    )
    {
        self.neural_net.add_input_edge(
            input_neuron_id, edge_id, input_array_idx, neg_weight, pos_weight, 
            self.input_rwlock_vec.clone(), 
            self.input_rwlock_grad_vec.clone()
        );
    }

    /// Add an edge for an output neuron to connect it to an index
    /// of the output array.
    pub fn add_output_edge(
        &mut self, output_neuron_id: &str, edge_id: String, 
        output_array_idx: usize, neg_weight: f32, pos_weight: f32,
    )
    {
        self.neural_net.add_output_edge(
            output_neuron_id, edge_id, output_array_idx, neg_weight, pos_weight, 
            self.output_rwlock_vec.clone(), 
            self.output_rwlock_grad_vec.clone()
        );
    }
}