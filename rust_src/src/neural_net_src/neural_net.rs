use std::{collections::HashMap, sync::{Arc, Mutex}};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::NeuronTrait};

/// Define the main neural network struct.
pub struct NeuralNet
{
    // Contains all the neurons and the edges that connect them.
    all_neurons: HashMap<String, Arc<Mutex<Box<dyn NeuronTrait>>>>,
    all_edges: HashMap<String, Arc<Mutex<Box<dyn EdgeTrait>>>>
}

impl NeuralNet
{
    pub fn new() -> Self
    {
        return Self
        {
            all_neurons: HashMap::new(),
            all_edges: HashMap::new()
        }
    }
}