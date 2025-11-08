use std::sync::{atomic::AtomicBool, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use crate::neural_net_src::{neuron_src::create_neuron::create_neuron, types_aliases::{ArcNeuronBufferVec, NeuronBuffer}};

/// Main function for threads to perform forward or backward
/// propagation of the neural network.
pub fn main_thread_fn(
    // Indicates whether threads should perform forward or backpropagation.
    traversal_mode: Arc<AtomicBool>,
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

    }
}