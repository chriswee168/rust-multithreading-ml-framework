use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

impl NeuralNetWrapper
{
    /// Join two neurons with a hidden edge.
    pub fn join_neurons(
        &mut self, neuron0_id: &str, neuron1_id: &str, 
        edge_id: String, neg_weight: f32, pos_weight: f32
    )
    {
        self.neural_net.join_neurons(neuron0_id, neuron1_id, edge_id, neg_weight, pos_weight);
    }
}