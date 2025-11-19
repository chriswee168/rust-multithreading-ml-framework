use std::{collections::{HashMap, VecDeque}, sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard}, thread::{self, JoinHandle}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, thread_src::main_thread_fn::main_thread_fn, types_aliases::{ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer}};

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

    // Required for direct memory transfer to thread buffers.
    pub forward_buffers: Vec<NeuronBuffer>,
    pub backward_buffers: Vec<NeuronBuffer>
    
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

            forward_buffers: Vec::new(),
            backward_buffers: Vec::new()
        }
    }

    /// Initialise the threads for neural net propagation.
    pub fn spawn_threads(
        &mut self, num_threads: usize,
        lr: f32,
        return_grad: bool
    )
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
                    i,
                    lr,
                    return_grad
                )
            );

            self.thread_handles.push(thread_handle);
        }
    }

    /// Initialise the forward buffer.
    pub fn init_forward_buffer(&mut self)
    {
        self.forward_buffers = self.init_neuron_buffer(&self.neural_net.input_neurons);
    }

    /// Initialize the backward buffer.
    pub fn init_backward_buffer(&mut self)
    {
        self.backward_buffers = self.init_neuron_buffer(&self.neural_net.output_neurons);
    }

    /// Private method for initialising the forward or backward buffers
    fn init_neuron_buffer(&self, io_neurons: &HashMap<String, ArcNeuronTrait>) -> Vec<NeuronBuffer>
    {
        let num_io_neurons: usize;
        num_io_neurons = io_neurons.len();

        let num_threads: usize = self.thread_handles.len();
        let neurons_per_buffer: usize = (num_io_neurons / num_threads) + 1;

        let mut increment: usize = 0;
        let mut buffer_guard_idx: usize = 0;
        
        // Create the buffers.
        let mut buffers: Vec<NeuronBuffer> = Vec::new();
        for _ in 0..self.thread_handles.len()
        {
            buffers.push(VecDeque::new());
        }

        // Select first buffer.
        let mut buffer: &mut NeuronBuffer = &mut buffers[buffer_guard_idx];

        // Fill each buffer with neurons_per_buffer neurons.
        for (_, neuron) in io_neurons
        {
            buffer.push_back(neuron.clone());
            increment += 1;

            if increment == neurons_per_buffer
            {
                // Obtain the next buffer.
                buffer_guard_idx += 1;
                buffer = &mut buffers[buffer_guard_idx];
                increment = 0;
            }
        }

        return buffers;
    }

    /// Set the traversal mode of the neural net. 
    /// (Either forward or backward propagation)
    pub fn prop_forward(&self, boolean: bool)
    {
        self.traverse_forward.store(boolean, Ordering::SeqCst);
    }

    /// Performs the full multi-threaded forward pass from input array to the output array.
    pub fn propagate(&self)
    {   
        let is_forward: bool = self.traverse_forward.load(Ordering::SeqCst);
        let edge_count: usize;
        let stored_buffers: &Vec<NeuronBuffer>;
        if is_forward
        {
            edge_count = self.neural_net.output_edges.len();
            stored_buffers = &self.forward_buffers;
        }
        else
        {
            edge_count = self.neural_net.input_edges.len();
            stored_buffers = &self.backward_buffers;
        }

        let thread_buffers: &ArcNeuronBufferVec = &self.thread_buffers;

        {
            // Initialize the output edge counter.
            let mut edge_count_guard: MutexGuard<'_, (usize, usize)> = self.edge_counter.1.lock().unwrap();
            edge_count_guard.0 = 0;
            edge_count_guard.1 = edge_count;
        }

        // Assign each stored buffer to thread buffer.
        for ((_, _, thread_buffer), stored_buffer) in 
            thread_buffers.iter().zip(stored_buffers)
        {
            let mut thread_buffer_guard: RwLockWriteGuard<'_, NeuronBuffer> = thread_buffer.write().unwrap();
            *thread_buffer_guard = stored_buffer.clone();
        }

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

    /// Display all neurons and their edges.
    pub fn display_params(&self)
    {
        for (neuron_id, neuron) in &self.neural_net.input_neurons
        {
            println!("neuron_id: {} | neuron_addr: {:p}", neuron_id, neuron);
            let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
            self.display_neuron_edges(neuron_guard);
            println!("----------");
        }

        for (neuron_id, neuron) in &self.neural_net.hidden_neurons
        {
            println!("neuron_id: {} | neuron_addr: {:p}", neuron_id, neuron);
            let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
            self.display_neuron_edges(neuron_guard);
            println!("----------");
        }

        for (neuron_id, neuron) in &self.neural_net.output_neurons
        {
            println!("neuron_id: {} | neuron_addr: {:p}", neuron_id, neuron);
            let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
            self.display_neuron_edges(neuron_guard);
            println!("----------");
        }
    }

    /// Displays the parameters of each edge in a neuron.
    fn display_neuron_edges(&self, neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>>)
    {
        // Display all backward edges.
        for (edge_id, edge) in neuron_guard.get_backward_edges()
        {
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            let edge_params: (f32, f32, f32) = edge_guard.get_params();
            println!("({}, {:p}) --> {}, {}, {}", edge_id, *edge, edge_params.0, edge_params.1, edge_params.2);
        }

        // Display all forward edges.
        for (edge_id, edge) in neuron_guard.get_forward_edges()
        {
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            let edge_params: (f32, f32, f32) = edge_guard.get_params();
            println!("{}, {}, {} --> ({}, {:p})", edge_params.0, edge_params.1, edge_params.2, edge_id, *edge);
        }
    }


}