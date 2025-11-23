use std::{collections::{HashMap, VecDeque}, ffi::c_void, hash::Hash, sync::MutexGuard};

use rand::Rng;

use crate::{neural_net_src::{edge_src::core_deps::EdgeTrait, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Randomly select an edge to remove if the absolute average of the edge parameters
/// is below a threshold.
#[unsafe(no_mangle)]
pub extern "C" fn remove_random_edge_ext(nn_vp: *mut c_void, edge_param_thresh: f32)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        let random_group: u32 = rand_gen.gen_range(0..=2);

        let mut edge: Option<ArcEdgeTrait> = None;
        let mut edge_id: Option<String> = None;

        match random_group
        {
            0 => {
                (edge_id, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.input_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            1 => {
                (edge_id, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.hidden_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            2 => {
                (edge_id, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.output_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            _ => ()
        }

        if edge.is_some()
        {
            remove_dead_ends(nn_ptr, edge_id.unwrap(), edge_param_thresh);
        }
    }
}

/// Obtain random edge from neural net.
fn obtain_random_edge(
    rand_gen: &mut rand::prelude::ThreadRng,
    edge_hashmap: &HashMap<String, ArcEdgeTrait>,
    neural_net: &NeuralNet
) -> (Option<String>, Option<ArcEdgeTrait>)
{
    let edge_vec: Vec<&String> = edge_hashmap.keys().collect();
    let random_idx: usize = rand_gen.gen_range(0..edge_vec.len());
    let random_edge_id: &String = edge_vec[random_idx];
    let (name, random_edge) = 
        neural_net.obtain_edge(&random_edge_id);
    return (Some(name.to_string()), random_edge);
}

/// Remove edges that don't provide direct pathways
/// from input to output (edges that lead to dead ends.)
pub fn remove_dead_ends(
    nn_ptr: *mut NeuralNetWrapper, 
    edge_id: String, 
    edge_param_thresh: f32
)
{
    let (_, edge) = 
        unsafe { (*nn_ptr).neural_net.obtain_edge(&edge_id) };

    let edge_arc: ArcEdgeTrait = edge.unwrap();
    let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
    let edge_params: (f32, f32, f32) = edge_guard.get_params();
    let param_abs_ave: f32 = 
        (edge_params.0.abs() + edge_params.1.abs() + edge_params.2.abs()) / 3.0;

    if param_abs_ave <= edge_param_thresh 
    {
        // Remove the randomly selected edge, and initialize the neuron buffer.
        let neuron1: Option<ArcNeuronTrait> = edge_guard.get_prev_neuron();
        let neuron2: Option<ArcNeuronTrait> = edge_guard.get_next_neuron();
        drop(edge_guard);

        unsafe { (*nn_ptr).remove_edge(&edge_id) };

        let mut neuron_buffer: VecDeque<Option<ArcNeuronTrait>> = VecDeque::from([neuron1, neuron2]);

        let mut edge_ids: Vec<&String>;

        // Traverse the network from the original two neurons connected by the
        // removed edge to disconnect dangling neurons.
        // (neurons that has zero backward or forward edges).
        while !neuron_buffer.is_empty() {
            let neuron: Option<ArcNeuronTrait> = neuron_buffer.pop_front().unwrap();
            if neuron.is_some() // Is a neuron and not an index.
            {
                let neuron_arc: ArcNeuronTrait = neuron.unwrap();
                let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron_arc.lock().unwrap();

                if neuron_guard.get_backward_edges().is_empty() || 
                    neuron_guard.get_forward_edges().is_empty()
                {
                    // Iterate through each forward and backward edge this neuron
                    // has, remove them and append the previous or next neurons to
                    // the buffer.
                            
                    edge_ids = neuron_guard.get_backward_edges().keys().collect();
                    for edge_id in edge_ids
                    {
                        let edge_arc: &ArcEdgeTrait = 
                            neuron_guard.get_backward_edges().get(edge_id).unwrap();
                        let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
                        let prev_neuron: Option<ArcNeuronTrait> = edge_guard.get_prev_neuron();
                        neuron_buffer.push_back(prev_neuron);
                        drop(edge_guard);

                        unsafe { (*nn_ptr).remove_edge(&edge_id) };
                    }

                    edge_ids = neuron_guard.get_forward_edges().keys().collect();
                    for edge_id in edge_ids
                    {
                        let edge_arc: &ArcEdgeTrait = 
                            neuron_guard.get_forward_edges().get(edge_id).unwrap();
                        let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
                        let prev_neuron: Option<ArcNeuronTrait> = edge_guard.get_next_neuron();
                        neuron_buffer.push_back(prev_neuron);
                        drop(edge_guard);

                        unsafe { (*nn_ptr).remove_edge(&edge_id) };
                    }
                }
            }
        }
    }
}