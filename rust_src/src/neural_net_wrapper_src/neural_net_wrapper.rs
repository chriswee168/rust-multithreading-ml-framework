use std::{sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard}, thread::{self, JoinHandle}};

use crate::neural_net_src::{neural_net::NeuralNet, thread_src::main_thread_fn::main_thread_fn, types_aliases::{ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer}};

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

    // Keeping track of input/output edges visited.
    // First usize is used as a counter.
    // Second usize is to keep the total number of inpt/output edges in the neural network.
    pub edge_counter: Arc<(Condvar, Mutex<(usize, usize)>)>,
    
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

            edge_counter: Arc::new((Condvar::new(), Mutex::new((0, 0)))),
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

    /// Performs the full multi-threaded forward pass from input array to the output array.
    pub fn forward(&self)
    {   
        // Divide number of input neurons by thread count to obtain 
        // the number of input neurons each thread should have.
        // Add one to round up.
        let num_input_neurons: usize = self.neural_net.input_neurons.len();
        let num_threads: usize = self.thread_handles.len();
        let neurons_per_buffer: usize = (num_input_neurons / num_threads) + 1;

        let mut increment: usize = 0;
        let mut buffer_guard_idx: usize = 0;
        let thread_buffers: &ArcNeuronBufferVec = &self.thread_buffers;

        {
            // Initialize the output edge counter.
            let mut edge_count_guard: MutexGuard<'_, (usize, usize)> = self.edge_counter.1.lock().unwrap();
            edge_count_guard.0 = 0;
            edge_count_guard.1 = self.neural_net.output_edges.len();
        }

        // Initialize first buffer write guard.
        let mut buffer_guard: RwLockWriteGuard<'_, NeuronBuffer> = (*thread_buffers)[buffer_guard_idx].2.write().unwrap();
        for (_, input_neuron) in &self.neural_net.input_neurons
        {
            buffer_guard.push_back(input_neuron.clone());
            increment += 1;

            if increment == neurons_per_buffer
            {
                // Obtain the write lock for the next thread's buffer.
                buffer_guard_idx += 1;
                buffer_guard = (*thread_buffers)[buffer_guard_idx].2.write().unwrap();
                increment = 0;
            }
        }

        // Explicitly drop the buffer write guard variable.
        drop(buffer_guard);

        // Notify all threads to begin BFS traversal on their own buffers.
        for buffer in thread_buffers.iter()
        {
            // Notify the thread that uses the current buffer to initiate BFS.
            let mut mutex_guard: MutexGuard<'_, bool> = buffer.1.lock().unwrap();
            *mutex_guard = true;
            buffer.0.notify_one();
        }

        // Wait on condvar to prevent this method from finishing before the 
        // neural network is fully traversed.
        let mut edge_count_guard: MutexGuard<'_, (usize, usize)> = self.edge_counter.1.lock().unwrap();
        // Ensure edge count is actually the same as the total number of output edges
        // to prevent spurious wakeups.
        while !(edge_count_guard.0 == edge_count_guard.1)
        {
            edge_count_guard = self.edge_counter.0.wait(edge_count_guard).unwrap();
        }
    }
}