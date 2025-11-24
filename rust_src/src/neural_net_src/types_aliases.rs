use std::{collections::VecDeque, sync::{Arc, Condvar, Mutex, RwLock}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::NeuronTrait};

// Type aliases for neurons.
pub type ArcNeuronTrait = Arc<Mutex<Box<dyn NeuronTrait>>>;

// Type aliases for edges.
pub type ArcEdgeTrait = Arc<Mutex<Box<dyn EdgeTrait>>>;

// Type aliases for neural network.
pub type NeuronBuffer = VecDeque<ArcNeuronTrait>; // Used to keep a buffer/queue of neurons for a CPU thread to work on.
// Contain a vector of neuron buffers for each CPU thread.
// Condvar and Mutex variables are needed for simulating a thread pool.
pub type ArcNeuronBufferVec = Arc<Vec<(Condvar, Mutex<bool>, RwLock<NeuronBuffer>)>>; 