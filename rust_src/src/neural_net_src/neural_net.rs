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
    pub fn new(num_threads: usize) -> Self
    {
        // Create thread buffers.
        let mut thread_buffers: Vec<Mutex<NeuronBuffer>> = Vec::with_capacity(num_threads);
        for _ in 0..num_threads - 1 // One thread is already used by main program.
        {
            thread_buffers.push(Mutex::new(NeuronBuffer::new()));
        }
        let thread_buffer_arc: ArcNeuronBufferVec = Arc::new(thread_buffers);
        
        // Create base neural network.
        return Self
        {
            all_neurons: HashMap::new(),
            all_edges: HashMap::new(),
            thread_buffers: thread_buffer_arc,
        }
    }
}