use std::{collections::{HashMap, VecDeque}, ffi::c_void, sync::MutexGuard};

use rand::Rng;

use crate::{neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::NeuronTrait, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

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

        let mut edge_id: Option<String> = None;

        match random_group
        {
            0 => {
                (edge_id, _) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.input_edges, 
                );
            }
            1 => {
                (edge_id, _) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.hidden_edges, 
                );
            }
            2 => {
               (edge_id, _) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.output_edges, 
                );
            }
            _ => ()
        }

        if edge_id.is_some()
        {
            remove_dead_ends(nn_ptr, edge_id.unwrap(), edge_param_thresh);
        }
    }
}

/// Obtain random edge from neural net.
pub fn obtain_random_edge(
    rand_gen: &mut rand::prelude::ThreadRng,
    edge_hashmap: &HashMap<String, ArcEdgeTrait>,
) -> (Option<String>, Option<ArcEdgeTrait>)
{
    let edge_vec: Vec<&String> = edge_hashmap.keys().collect();
    if !edge_vec.is_empty()
    {
        let random_idx: usize = rand_gen.gen_range(0..edge_vec.len());
        let random_edge_id: &String = edge_vec[random_idx];
        let random_edge: ArcEdgeTrait = edge_hashmap.get(random_edge_id).unwrap().clone();
        return (Some(random_edge_id.to_string()), Some(random_edge));
    }
    else
    {
        return (None, None);
    }
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
    let edge_params: (f32, f32) = edge_guard.get_params();
    let param_abs_ave: f32 = (edge_params.0.abs() + edge_params.1.abs()) / 2.0;

    if param_abs_ave <= edge_param_thresh 
    {
        // Remove the randomly selected edge, and initialize the neuron buffer.
        let neuron1: Option<ArcNeuronTrait> = edge_guard.get_prev_neuron();
        let neuron2: Option<ArcNeuronTrait> = edge_guard.get_next_neuron();
        drop(edge_guard);

        unsafe { (*nn_ptr).remove_edge(&edge_id) };

        let mut neuron_buffer: VecDeque<Option<ArcNeuronTrait>> = VecDeque::from([neuron1, neuron2]);
        
        // Contains IDs of hidden neurons that have no edges.
        let mut neurons_to_remove: Vec<String> = Vec::new();

        // Traverse the network from the original two neurons connected by the
        // removed edge to disconnect dangling neurons.
        // (neurons that has zero backward or forward edges).
        while !neuron_buffer.is_empty() {
            let neuron: Option<ArcNeuronTrait> = neuron_buffer.pop_front().unwrap();
            if neuron.is_some() // Is a neuron and not an index.
            {
                let neuron_arc: ArcNeuronTrait = neuron.unwrap();
                let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron_arc.lock().unwrap();
                let is_dangling: bool = neuron_guard.get_backward_edges().is_empty() || 
                    neuron_guard.get_forward_edges().is_empty();
                
                // Obtain the IDs of backward and forward edges and drop the neuron mutex
                // guard to prevent deadlocks.
                let backward_edge_ids: Vec<&String> = neuron_guard.get_backward_edges().keys().collect();
                let forward_edge_ids: Vec<&String> = neuron_guard.get_forward_edges().keys().collect();
                let backward_edge_ids: Vec<String> = backward_edge_ids.iter().map(|id: &&String| id.to_string()).collect();
                let forward_edge_ids: Vec<String> = forward_edge_ids.iter().map(|id: &&String| id.to_string()).collect();
                let neuron_id: String = neuron_guard.get_neuron_id();
                let is_hidden_neuron: bool = neuron_guard.get_neuron_level().is_some();
                drop(neuron_guard);
                
                if is_dangling
                {
                    // Iterate through each forward and backward edge this neuron
                    // has, remove them and append the previous or next neurons to
                    // the buffer.

                    for edge_id in backward_edge_ids
                    {
                        let edge_arc: ArcEdgeTrait = 
                            unsafe { (*nn_ptr).neural_net.obtain_edge(&edge_id).1.unwrap() };
                        let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
                        let prev_neuron: Option<ArcNeuronTrait> = edge_guard.get_prev_neuron();
                        neuron_buffer.push_back(prev_neuron);
                        drop(edge_guard);

                        unsafe { (*nn_ptr).remove_edge(&edge_id) };
                    }

                    for edge_id in forward_edge_ids
                    {
                        let edge_arc: ArcEdgeTrait = 
                            unsafe { (*nn_ptr).neural_net.obtain_edge(&edge_id).1.unwrap() };
                        let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
                        let prev_neuron: Option<ArcNeuronTrait> = edge_guard.get_next_neuron();
                        neuron_buffer.push_back(prev_neuron);
                        drop(edge_guard);

                        unsafe { (*nn_ptr).remove_edge(&edge_id) };
                    }

                    if is_hidden_neuron
                    {
                        neurons_to_remove.push(neuron_id);
                    }
                }
            }
        }

        // Remove each hidden neuron that had its edges removed.
        for neuron_id in neurons_to_remove
        {
            unsafe { (*nn_ptr).neural_net.remove_empty_neuron(&neuron_id); }
        }
    }
}