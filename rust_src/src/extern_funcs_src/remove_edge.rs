use std::{collections::{HashMap, VecDeque}, ffi::c_void, hash::Hash, sync::MutexGuard};

use rand::Rng;

use crate::{neural_net_src::{edge_src::core_deps::EdgeTrait, neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

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
            let edge_arc: ArcEdgeTrait = edge.unwrap();
            let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge_arc.lock().unwrap();
            let edge_params: (f32, f32, f32) = edge_guard.get_params();
            let param_abs_ave: f32 = 
                (edge_params.0.abs() + edge_params.1.abs() + edge_params.2.abs()) / 3.0;

            if param_abs_ave <= edge_param_thresh 
            {
                (*nn_ptr).remove_edge(&edge_id.unwrap());
            }
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