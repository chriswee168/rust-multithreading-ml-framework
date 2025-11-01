use crate::neural_net_src::{neural_net::NeuralNet, types_aliases::ArcEdgeTrait};

impl NeuralNet
{
    /// Obtain the edge arc from either input, hidden or output edge hashmap.
    pub fn obtain_edge(&self, edge_id: String) -> (&str, Option<ArcEdgeTrait>)
    {
        if self.input_edges.get(&edge_id).is_some()
        {
            return ("input", Some(self.input_edges.get(&edge_id).unwrap().clone()));
        }
        else if self.hidden_edges.get(&edge_id).is_some() 
        {
            return ("hidden", Some(self.hidden_edges.get(&edge_id).unwrap().clone()));
        }
        else if self.output_edges.get(&edge_id).is_some()
        {
            return ("output", Some(self.output_edges.get(&edge_id).unwrap().clone()));
        }
        else
        {
            ("none", None)
        }
    }
}