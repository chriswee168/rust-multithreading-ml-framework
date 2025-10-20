use crate::neural_net_src::edge_src::direct_edge::DirectEdge;

/// Struct that contains all attributes required for input,
/// hidden and output neurons.
pub struct NeuronAttr<'a>
{
    forward_edges: Vec<&'a mut DirectEdge>,
    backward_edges: Vec<&'a mut DirectEdge>,

    neuron_level: u32,
    neuron_id: String,
}
