use std::{collections::HashMap, ffi::c_void, sync::MutexGuard};

use rand::Rng;

use crate::{neural_net_src::{neural_net::NeuralNet, neuron_src::core_deps::NeuronTrait, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Randomly select a neuron that has no edges.
#[unsafe(no_mangle)]
pub extern "C" fn remove_random_neuron_ext(nn_vp: *mut c_void)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        let random_group: u32 = rand_gen.gen_range(0..=2);

        let mut neuron_id: Option<String> = None;

        match random_group
        {
            0 => {
                neuron_id = obtain_random_neuron(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.input_neurons, 
                );
            }
            1 => {
                neuron_id = obtain_random_neuron(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.hidden_neurons, 
                );
            }
            2 => {
                neuron_id = obtain_random_neuron(
                    &mut rand_gen, 
                    &(*nn_ptr).neural_net.output_neurons, 
                );
            }
            _ => ()
        }

        if neuron_id.is_some()
        {
            (*nn_ptr).neural_net.remove_empty_neuron(&neuron_id.unwrap());
        }
    }
}

/// Obtain random neuron from neural net.
fn obtain_random_neuron(
    rand_gen: &mut rand::prelude::ThreadRng,
    neuron_hashmap: &HashMap<String, ArcNeuronTrait>,
) -> Option<String>
{
    let neuron_vec: Vec<&String> = neuron_hashmap.keys().collect();
    if !neuron_vec.is_empty()
    {
        let random_idx: usize = rand_gen.gen_range(0..neuron_vec.len());
        let random_neuron_id: &String = neuron_vec[random_idx];
        return Some(random_neuron_id.to_string());
    }
    else
    {
        return None;
    }
}