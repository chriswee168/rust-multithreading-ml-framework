use std::{collections::HashMap, ffi::{c_char, c_void, CStr}, sync::MutexGuard};

use rand::Rng;

use crate::{neural_net_src::{neuron_src::core_deps::NeuronTrait, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Join two neurons together using a hidden edge.
#[unsafe(no_mangle)]
pub extern "C" fn join_two_rand_neurons_ext(
    nn_vp: *mut c_void, neg_weight: f32, pos_weight: f32,
    neuron_group1: usize, neuron_group2: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        let (mut neuron_id1, mut neuron_id2) = (None, None);
        let (mut neuron1, mut neuron2) = (None, None);

        match (neuron_group1, neuron_group2)
        {
            // Input neuron and hidden neuron.
            (0, 1) => {
                (neuron_id1, neuron1) = get_rand_neuron(
                    &(*nn_ptr).neural_net.input_neurons, 
                    &mut rand_gen
                );

                (neuron_id2, neuron2) = get_rand_neuron(
                    &(*nn_ptr).neural_net.hidden_neurons, 
                    &mut rand_gen
                );
            }
            // Two hidden neurons.
            (1, 1) => {
                (neuron_id1, neuron1) = get_rand_neuron(
                    &(*nn_ptr).neural_net.hidden_neurons, 
                    &mut rand_gen
                );

                (neuron_id2, neuron2) = get_rand_neuron(
                    &(*nn_ptr).neural_net.hidden_neurons, 
                    &mut rand_gen
                );
            }
            // Hidden neuron and output neuron.
            (1, 2) => {
                (neuron_id1, neuron1) = get_rand_neuron(
                    &(*nn_ptr).neural_net.hidden_neurons, 
                    &mut rand_gen
                );

                (neuron_id2, neuron2) = get_rand_neuron(
                    &(*nn_ptr).neural_net.output_neurons, 
                    &mut rand_gen
                );
            }
            // Input neuron and output neuron.
            (0, 2) => {
                (neuron_id1, neuron1) = get_rand_neuron(
                    &(*nn_ptr).neural_net.input_neurons, 
                    &mut rand_gen
                );

                (neuron_id2, neuron2) = get_rand_neuron(
                    &(*nn_ptr).neural_net.output_neurons, 
                    &mut rand_gen
                );
            }

            // No other combination if accepted to prevent cyclic
            // edge connections.
            _ => ()
        }
        
        if neuron_id1.is_some() && neuron_id2.is_some() && neuron_id1 != neuron_id2
        {
            let neuron1: ArcNeuronTrait = neuron1.unwrap();
            let neuron2: ArcNeuronTrait = neuron2.unwrap();
            let neuron1_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron1.lock().unwrap();
            let neuron2_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron2.lock().unwrap();
            
            let mut ascending_levels: bool = true;
            let neuron1_level: Option<u32> = neuron1_guard.get_neuron_level();
            let neuron2_level: Option<u32> = neuron2_guard.get_neuron_level();
            
            // Only if both neuron1 and neuron2 are hidden neurons.
            if (neuron1_level.is_some() && neuron2_level.is_some()) && neuron1_level >= neuron2_level
            {
                ascending_levels = false
            }

            let not_exceeded_max_edges: bool = 
                neuron1_guard.get_forward_edges().len() < neuron1_guard.get_forward_edge_max() &&
                neuron2_guard.get_backward_edges().len() < neuron2_guard.get_backward_edge_max();

            // Drop neuron guards.
            drop(neuron1_guard);
            drop(neuron2_guard);

            if ascending_levels && not_exceeded_max_edges
            {
                let edge_id: String = 
                    neuron_id1.clone().unwrap() + " --> " + neuron_id2.clone().unwrap().as_str();

                (*nn_ptr).join_neurons(
                    &neuron_id1.unwrap(), &neuron_id2.unwrap(), 
                    edge_id, neg_weight, pos_weight
                );
            }
        }
    }
}

/// Get random neuron and its ID from neuron group.
pub fn get_rand_neuron(
    neuron_group: &HashMap<String, ArcNeuronTrait>,
    rand_gen: &mut rand::prelude::ThreadRng
) -> (Option<String>, Option<ArcNeuronTrait>)
{
    let neuron_group_keys: Vec<&String> = neuron_group.keys().collect();

    if neuron_group_keys.len() > 0
    {
        let rand_idx: usize = rand_gen.gen_range(0..neuron_group_keys.len());
        let neuron_id: String = neuron_group_keys[rand_idx].clone();
        let neuron: Option<ArcNeuronTrait> = neuron_group.get(&neuron_id).cloned();

        return (Some(neuron_id), neuron);
    }
    else
    {
        return (None, None)
    }
}

/// Add an input edge to an input neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_input_edge_ext(
    nn_vp: *mut c_void, input_neuron_id: *mut c_char, arr_idx: usize,
    neg_weight: f32, pos_weight: f32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let neuron_id_str: &str = CStr::from_ptr(input_neuron_id).to_str().unwrap();
        let edge_id: String = arr_idx.to_string() + " --> " + neuron_id_str;

        (*nn_ptr).add_input_edge(
            neuron_id_str, edge_id, 
            arr_idx, neg_weight, pos_weight
        );
    }
}


/// Add an output edge to an output neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_output_edge_ext(
    nn_vp: *mut c_void, output_neuron_id: *mut c_char, arr_idx: usize,
    neg_weight: f32, pos_weight: f32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let neuron_id_str: &str = CStr::from_ptr(output_neuron_id).to_str().unwrap();
        let edge_id: String = neuron_id_str.to_string() +  "--> " + arr_idx.to_string().as_str();

        (*nn_ptr).add_output_edge(
            neuron_id_str, edge_id, 
            arr_idx, neg_weight, pos_weight
        );
    }
}