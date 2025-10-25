use std::{collections::HashMap};

use crate::neural_net_src::{types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

/// Define the main neural network struct.
pub struct NeuralNet
{
    // Contains all the neurons and the edges that connect them.
    all_neurons: HashMap<String, ArcNeuronTrait>,
    all_edges: HashMap<String, ArcEdgeTrait>
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