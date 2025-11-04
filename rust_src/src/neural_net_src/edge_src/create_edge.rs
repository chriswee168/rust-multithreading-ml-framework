use std::sync::{Arc, Mutex};

use crate::neural_net_src::{
    edge_src::{element_mutexed_vec::ElementMutexedVec, 
        hidden_edge::HiddenEdge, input_edge::InputEdge, 
        output_edge::OutputEdge}, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

/// Create input edge.
pub fn create_input_edge(
    input_array_idx: usize, input_neuron: &ArcNeuronTrait, 
    input_mutexed_vec: Arc<ElementMutexedVec<f32>>, neg_weight: f32, pos_weight: f32
) -> ArcEdgeTrait
{
    let input_edge: ArcEdgeTrait = 
        Arc::new(
            Mutex::new(
                Box::new(
                    InputEdge::new(
                        input_array_idx, 
                        input_neuron.clone(), 
                        input_mutexed_vec.clone(), neg_weight, pos_weight
                    )
                )
            )
        );
    
    return input_edge;
}

/// Create output edge.
pub fn create_output_edge(
    output_neuron: &ArcNeuronTrait, output_array_idx: usize, 
    output_mutexed_vec: Arc<ElementMutexedVec<f32>>, neg_weight: f32, pos_weight: f32
) -> ArcEdgeTrait
{
    let output_edge: ArcEdgeTrait = 
        Arc::new(
            Mutex::new(
                Box::new(
                    OutputEdge::new(
                        output_neuron.clone(),
                        output_array_idx,  
                        output_mutexed_vec.clone(), neg_weight, pos_weight
                    )
                )
            )
        );
    
    return output_edge;
}

/// Create hidden edge.
pub fn create_hidden_edge(
    neuron0: &ArcNeuronTrait, neuron1: &ArcNeuronTrait, neg_weight: f32, pos_weight: f32
) -> ArcEdgeTrait
{
    let hidden_edge: ArcEdgeTrait = 
        Arc::new(
            Mutex::new(
                Box::new(
                    HiddenEdge::new(
                        neuron0.clone(),
                        neuron1.clone(),  
                        neg_weight, pos_weight
                    )
                )
            )
        );
    
    return hidden_edge;
}