use std::{collections::{HashMap}, sync::{Arc, Mutex}};

use crate::neural_net_src::types_aliases::{ArcEdgeTrait, ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer};

/// Define the main neural network struct.
pub struct NeuralNet
{
    // Contains an Arc reference to all neurons and edges in the
    // neural net.
    all_neurons: HashMap<String, ArcNeuronTrait>,
    all_edges: HashMap<String, ArcEdgeTrait>,

    // Neuron buffers are used for Breadth First Search traversal
    // during forward and backward propagation through neurons/edges.
    // Each CPU thread uses its own buffer to reduce contention.
    thread_buffers: ArcNeuronBufferVec
}

impl NeuralNet
{
    pub fn new() -> Self
    {
        return Self
        {
            all_neurons: HashMap::new(),
            all_edges: HashMap::new()
        }
    }
}