use std::sync::{atomic::{AtomicBool, Ordering}, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use crate::neural_net_src::{neuron_src::{core_deps::NeuronTrait, create_neuron::create_neuron}, types_aliases::{ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer}};

/// Main function for threads to perform forward or backward
/// propagation of the neural network.
pub fn main_thread_fn(
    // Indicates whether threads should perform forward or backpropagation.
    traverse_forward: Arc<AtomicBool>,
    // Arc pointer to vector containing all neuron buffers for every thread.
    neuron_buffers: ArcNeuronBufferVec, 
    // Specific index of the neuron buffer to use for this thread.
    buffer_idx: usize
)
{
    // Get specific neuron buffer for this thread as well as its condvar and mutex.
    let tuple: &(Condvar, Mutex<bool>, RwLock<NeuronBuffer>) = 
        &neuron_buffers[buffer_idx];

    // Get writer lock for this threads neuron buffer.
    let mut buffer_guard: RwLockWriteGuard<'_, NeuronBuffer> = tuple.2.write().unwrap();
    
    // Temporary for loop.
    for i in 0..100
    {
        {
            // Wait on Condvar, this thread will be notified and woken up if Mutex
            // is true to indicate neuron/s are present in its queue to pop off and
            // work on.
            let mut mutex_guard: MutexGuard<'_, bool> = tuple.1.lock().unwrap();
            while !*mutex_guard // Prevent spurious wakeups.
            {
                mutex_guard = tuple.0.wait(mutex_guard).unwrap();
            }
        }

        let traversal_bool: bool = traverse_forward.load(Ordering::SeqCst);
        
        // Start Breadth First Search traversal.
        while !buffer_guard.is_empty()
        {
            // Remove first neuron from buffer.
            let neuron: ArcNeuronTrait = buffer_guard.pop_front().unwrap();
            let mut neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();

            // Check if neuron's visit count is the same the number of edges depending
            // on traversal mode.
            // Neurons only propagate values if it has received total sum from all
            // previous edges. 
            let n_edges: usize;
            let visit_count: usize = neuron_guard.get_visit_count(traversal_bool);
            if traversal_bool // Perform forward propagation.
            {
                n_edges = neuron_guard.get_backward_edges().len();
                if n_edges == visit_count
                {
                    neuron_guard.forward();
                }
            }
            else // Perform backpropagation.
            {
                n_edges = neuron_guard.get_forward_edges().len();
                if n_edges == visit_count
                {
                    neuron_guard.backward();
                }
            }
        }
    }
}