use std::sync::{atomic::AtomicBool, Arc};

use crate::neural_net_src::types_aliases::ArcNeuronBufferVec;

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
    
}