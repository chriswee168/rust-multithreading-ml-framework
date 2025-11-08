use std::{collections::HashMap, sync::{Arc, Condvar, Mutex, RwLock}, thread::JoinHandle};

use crate::neural_net_src::types_aliases::{ArcEdgeTrait, ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer};

/// Define the main neural network struct.
pub struct NeuralNet
{
    // Contains Arc references to input, hidden and output neurons.
    pub input_neurons: HashMap<String, ArcNeuronTrait>,
    pub hidden_neurons: HashMap<String, ArcNeuronTrait>,
    pub output_neurons: HashMap<String, ArcNeuronTrait>,

    pub input_edges: HashMap<String, ArcEdgeTrait>, // input array -> input neurons
    pub hidden_edges: HashMap<String, ArcEdgeTrait>, // neuron -> neuron
    pub output_edges: HashMap<String, ArcEdgeTrait>, // output neuron -> output array

    // Neuron buffers are used for Breadth First Search traversal
    // during forward and backward propagation through neurons/edges.
    // Each CPU thread uses its own buffer to reduce contention.
    pub thread_buffers: ArcNeuronBufferVec,

    // Contains handles of each thread for thread management.
    pub thread_handles: Vec<JoinHandle<()>>
}

impl NeuralNet
{
    pub fn new(num_threads: usize) -> Self
    {
        // Create thread buffers.
        let mut thread_buffers: Vec<(Condvar, Mutex<bool>, RwLock<NeuronBuffer>)> = 
            Vec::with_capacity(num_threads);

        for _ in 0..num_threads - 1 // One thread is already used by main program.
        {
            thread_buffers.push((
                Condvar::new(),
                Mutex::new(false),
                RwLock::new(NeuronBuffer::new())
            ));
        }
        
        let thread_buffer_arc: ArcNeuronBufferVec = Arc::new(thread_buffers);
        
        // Create base neural network.
        return Self
        {
            input_neurons: HashMap::new(),
            hidden_neurons: HashMap::new(),
            output_neurons: HashMap::new(),

            input_edges: HashMap::new(),
            hidden_edges: HashMap::new(),
            output_edges: HashMap::new(),
            
            thread_buffers: thread_buffer_arc,
            thread_handles: Vec::new()
        }
    }
}