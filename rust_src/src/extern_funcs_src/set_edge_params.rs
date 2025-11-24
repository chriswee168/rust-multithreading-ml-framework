use std::{ffi::c_void, sync::MutexGuard};

use rand::Rng;

use crate::{extern_funcs_src::remove_edge::obtain_random_edge, neural_net_src::{edge_src::core_deps::EdgeTrait, types_aliases::ArcEdgeTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Change the weight parameters of a random input, hidden or output edge.
#[unsafe(no_mangle)]
pub extern "C" fn set_rand_edge_params(
    nn_vp: *mut c_void,
    pos_param: f32,
    neg_param: f32,
)
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
                (_, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.input_edges, 
                );
            }
            1 => {
                (_, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.hidden_edges, 
                );
            }
            2 => {
                (_, edge) = obtain_random_edge(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.output_edges, 
                );
            }
            _ => ()
        }

        
        if edge.is_some()
        {
            let edge: ArcEdgeTrait = edge.unwrap();
            let mut edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
            edge_guard.set_params(pos_param, neg_param);
        }
    }    
}