use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

impl NeuralNetWrapper
{
    /// Remove an edge.
    pub fn remove_edge(&mut self, edge_id: &str)
    {
        self.neural_net.remove_edge(edge_id);
    }
}