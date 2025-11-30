use std::collections::HashMap;
use crate::neural_net_src::types_aliases::{ArcEdgeTrait, ArcNeuronTrait};

/// Define the main neural network struct.
pub struct NeuralNet
{
    // Contains Arc references to input, hidden and output neurons.
    pub input_neurons: HashMap<String, ArcNeuronTrait>,
    pub hidden_neurons: HashMap<String, ArcNeuronTrait>,
    pub output_neurons: HashMap<String, ArcNeuronTrait>,

    pub input_edges: HashMap<String, ArcEdgeTrait>, // input array -> input neurons
    pub hidden_edges: HashMap<String, ArcEdgeTrait>, // neuron -> neuron
    pub output_edges: HashMap<String, ArcEdgeTrait>, // output neuron -> output array
}

impl NeuralNet
{
    pub fn new() -> Self
    {           
        // Create base neural network.
        return Self
        {
            input_neurons: HashMap::new(),
            hidden_neurons: HashMap::new(),
            output_neurons: HashMap::new(),

            input_edges: HashMap::new(),
            hidden_edges: HashMap::new(),
            output_edges: HashMap::new(),
        }
    }
}