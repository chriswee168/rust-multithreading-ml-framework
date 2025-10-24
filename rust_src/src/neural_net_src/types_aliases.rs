use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::NeuronTrait};

// Type aliases for neurons.
pub type ArcNeuronTrait = Arc<Mutex<Box<dyn NeuronTrait>>>;

// Type aliases for edges.
pub type ArcEdgeTrait<T, U> = Arc<Mutex<Box<dyn EdgeTrait<T, U>>>>;