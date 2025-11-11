use std::{sync::{atomic::AtomicBool, Arc, RwLock}, thread::JoinHandle};

use crate::neural_net_src::{neural_net::NeuralNet, types_aliases::ArcNeuronBufferVec};

pub struct NeuralNetWrapper
{
    // Struct of the neural network.
    neural_net: NeuralNet,

    // Neuron buffers are used for Breadth First Search traversal
    // during forward and backward propagation through neurons/edges.
    // Each CPU thread uses its own buffer to reduce contention.
    thread_buffers: ArcNeuronBufferVec,

    // Contains handles of each thread for thread management.
    thread_handles: Vec<JoinHandle<()>>,

    // Atomic boolean to indicate whether threads should perform forward
    // or backpropagation. (true by default)
    traverse_forward: Arc<AtomicBool>,

    // RwLocked input vectors for neural network to process
    // and RwLocked output vectors to store outputs.
    input_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    input_rwlock_grad_vec: Arc<RwLock<Vec<f32>>>,
    output_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    output_rwlock_grad_vec: Arc<RwLock<Vec<f32>>>,
    
}
impl NeuralNetWrapper
{
    pub fn new() -> Self
    {   
        let thread_buffer_arc: ArcNeuronBufferVec = Arc::new(Vec::new());

        return Self
        {
            neural_net: NeuralNet::new(),
            
            thread_buffers: thread_buffer_arc,
            thread_handles: Vec::new(),
            traverse_forward: Arc::new(AtomicBool::new(true)),

            input_rwlock_vec: Arc::new(RwLock::new(Vec::new())),
            input_rwlock_grad_vec: Arc::new(RwLock::new(Vec::new())),
            output_rwlock_vec: Arc::new(RwLock::new(Vec::new())),
            output_rwlock_grad_vec: Arc::new(RwLock::new(Vec::new())),
        }
    }
}