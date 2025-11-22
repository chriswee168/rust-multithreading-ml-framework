use std::ffi::c_void;

use rand::Rng;

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

#[unsafe(no_mangle)]
pub extern "C" fn remove_random_edge_ext(nn_vp: *mut c_void, edge_param_thresh: f32)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        let random_group: u32 = rand_gen.gen_range(0..=2);

        let mut edge_ids: Option<Vec<&String>> = None;

        match random_group
        {
            0 => {
                edge_ids = Some((*nn_ptr).neural_net.input_edges.keys().collect());
            }
            1 => {
                edge_ids = Some((*nn_ptr).neural_net.hidden_edges.keys().collect());
            }
            2 => {
                edge_ids = Some((*nn_ptr).neural_net.output_edges.keys().collect());
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