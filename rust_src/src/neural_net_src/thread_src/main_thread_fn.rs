use std::sync::{atomic::{AtomicBool, Ordering}, Arc, Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use crate::neural_net_src::{neuron_src::{core_deps::NeuronTrait}, types_aliases::{ArcNeuronBufferVec, ArcNeuronTrait, NeuronBuffer}};

/// Main function for threads to perform forward or backward
/// propagation of the neural network.
pub fn main_thread_fn(
    // Indicates whether threads should perform forward or backpropagation.
    traverse_forward: Arc<AtomicBool>,
    // Arc pointer to vector containing all neuron buffers for every thread.
    neuron_buffers: ArcNeuronBufferVec, 
    // Specific index of the neuron buffer to use for this thread.
    buffer_idx: usize,
    // Learning rate for gradient descent.
    lr: f32,
    // Choose whether the neural net will return the final
    // gradients of length input dim.
    return_grads: bool,
    // Used for the last thread that has no neurons left to work on
    // to unpause the main propagation method.
    threads_finished: Arc<(Condvar, Mutex<(usize, usize)>)>
)
{
    // Get specific neuron buffer for this thread as well as its condvar and mutex.
    let tuple: &(Condvar, Mutex<bool>, RwLock<NeuronBuffer>) = 
        &neuron_buffers[buffer_idx];
    
    // Acquire mutex guard for condvar wait.
    let mut mutex_guard: MutexGuard<'_, bool> = tuple.1.lock().unwrap();
    
    loop
    {
        // Wait on Condvar, this thread will be notified and woken up if Mutex
        // is true to indicate neuron/s are present in its queue to pop off and
        // work on.
        while !*mutex_guard // Prevent spurious wakeups.
        {
            mutex_guard = tuple.0.wait(mutex_guard).unwrap();
        }

        let traversal_bool: bool = traverse_forward.load(Ordering::SeqCst);

        {
            // Get writer lock for this threads neuron buffer.
            let mut buffer_guard: RwLockWriteGuard<'_, NeuronBuffer> = tuple.2.write().unwrap();

            // Start Breadth First Search traversal.
            while !buffer_guard.is_empty()
            {
                // Remove first neuron from buffer.
                let neuron: ArcNeuronTrait = buffer_guard.pop_front().unwrap();
                let mut neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();

                if traversal_bool // Perform forward propagation.
                {
                    neuron_guard.forward(&mut buffer_guard);
                }
                else // Perform backpropagation.
                {
                    neuron_guard.backward(lr, return_grads, &mut buffer_guard);
                }
            }
        }

        // Reset mutex guard to false to block thread at condvar.
        *mutex_guard = false;

        {
            // Increment the threads_finished counter as this thread no longer
            // has any neurons to work on.
            let mut counter_guard: MutexGuard<'_, (usize, usize)> = threads_finished.1.lock().unwrap();
            counter_guard.0 += 1;

            // Notify main thread when all threads have no neurons to work on.
            if counter_guard.0 == counter_guard.1
            {
                threads_finished.0.notify_one();
            }
        }
    }
}