use std::sync::{Arc, Mutex};

use crate::neural_net_src::{edge_src::{element_mutexed_vec::ElementMutexedVec, input_edge::InputEdge}, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

/// Create input edge.
pub fn create_input_edge(
    input_array_idx: usize, input_neuron: ArcNeuronTrait, 
    input_mutexed_vec: Arc<ElementMutexedVec<f32>>, edge_weight_range: f32 
) -> ArcEdgeTrait
{
    let input_edge: ArcEdgeTrait = 
        Arc::new(
            Mutex::new(
                Box::new(
                    InputEdge::new(
                        input_array_idx, 
                        input_neuron.clone(), 
                        input_mutexed_vec.clone(), edge_weight_range
                    )
                )
            )
        );
    
    return input_edge;
}