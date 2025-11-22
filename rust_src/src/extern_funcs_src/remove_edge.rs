use std::{collections::HashMap, ffi::c_void, hash::Hash};

use rand::Rng;

use crate::{neural_net_src::{neural_net::NeuralNet, types_aliases::ArcEdgeTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

#[unsafe(no_mangle)]
pub extern "C" fn remove_random_edge_ext(nn_vp: *mut c_void, edge_param_thresh: f32)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        let random_group: u32 = rand_gen.gen_range(0..=2);

        let mut edge: Option<ArcEdgeTrait> = None;

        match random_group
        {
            0 => {
                edge = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.input_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            1 => {
                edge = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.hidden_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            2 => {
                edge = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.output_edges, 
                    &(*nn_ptr).neural_net
                );
            }
            _ => ()
        }

        if edge_ids.is_some()
        {
            let edge_vec: Vec<&String> = edge_ids.unwrap();
            let random_idx: usize = rand_gen.gen_range(0..edge_vec.len());
            let random_edge_id: &String = edge_vec[random_idx];
            let (_, edge) = 
                (*nn_ptr).neural_net.obtain_edge(&random_edge_id);
            let edge_arc = edge.unwrap();
        }
    }
}

/// Obtain random edge from neural net.
fn obtain_random_edge(
    rand_gen: &mut rand::prelude::ThreadRng,
    edge_hashmap: &HashMap<String, ArcEdgeTrait>,
    neural_net: &NeuralNet
) -> Option<ArcEdgeTrait>
{
    let edge_vec: Vec<&String> = edge_hashmap.keys().collect();
    let random_idx: usize = rand_gen.gen_range(0..edge_vec.len());
    let random_edge_id: &String = edge_vec[random_idx];
    let (_, random_edge) = 
        neural_net.obtain_edge(&random_edge_id);
    return random_edge;
}