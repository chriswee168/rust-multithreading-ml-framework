use std::{sync::{atomic::{AtomicBool, Ordering}, Arc, Condvar, Mutex, RwLock}, thread::{self, JoinHandle}};

use crate::neural_net_src::{neural_net::NeuralNet, thread_src::main_thread_fn::main_thread_fn, types_aliases::{ArcNeuronBufferVec, NeuronBuffer}};

pub struct NeuralNetWrapper
{
    // Struct of the neural network.
    pub neural_net: NeuralNet,

    // Neuron buffers are used for Breadth First Search traversal
    // during forward and backward propagation through neurons/edges.
    // Each CPU thread uses its own buffer to reduce contention.
    pub thread_buffers: ArcNeuronBufferVec,

    // Contains handles of each thread for thread management.
    pub thread_handles: Vec<JoinHandle<()>>,

    // Atomic boolean to indicate whether threads should perform forward
    // or backpropagation. (true by default)
    pub traverse_forward: Arc<AtomicBool>,

    // RwLocked input vectors for neural network to process
    // and RwLocked output vectors to store outputs.
    pub input_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    pub input_rwlock_grad_vec: Arc<RwLock<Vec<f32>>>,
    pub output_rwlock_vec: Arc<RwLock<Vec<f32>>>,
    pub output_rwlock_grad_vec: Arc<RwLock<Vec<f32>>>,
    
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

    /// Initialise the threads for neural net propagation.
    pub fn spawn_threads(&mut self, num_threads: usize)
    {
        // Create thread buffers.
        let mut thread_buffers: Vec<(Condvar, Mutex<bool>, RwLock<NeuronBuffer>)> = 
            Vec::with_capacity(num_threads);

        for _ in 0..num_threads
        {
            thread_buffers.push((
                Condvar::new(),
                Mutex::new(false),
                RwLock::new(NeuronBuffer::new())
            ));
        }

        // Re-assign number of thread buffers.
        self.thread_buffers = Arc::new(thread_buffers);

        // Clear thread handles.
        self.thread_handles.clear();

        // Spawn threads and keep their handles.
        for i in 0..num_threads
        {
            let traverse_forward_clone: Arc<AtomicBool> = self.traverse_forward.clone();
            let thread_buffer_clone: ArcNeuronBufferVec = self.thread_buffers.clone();
            
            let thread_handle: JoinHandle<()> = thread::spawn(
                move || main_thread_fn(
                    traverse_forward_clone, 
                    thread_buffer_clone, 
                    i
                )
            );

            self.thread_handles.push(thread_handle);
        }
    }

    /// Set the traversal mode of the neural net. 
    /// (Either forward or backward propagation)
    pub fn prop_forward(&self, boolean: bool)
    {
        self.traverse_forward.store(boolean, Ordering::SeqCst);
    }
}