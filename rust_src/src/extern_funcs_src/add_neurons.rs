use std::{ffi::{c_char, c_void, CStr}, u32::MAX};

use crate::{neural_net_src::{
    neuron_src::create_neuron::create_neuron, rand_id_gen::rand_id_gen, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Add input neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_input_neuron_ext(
    nn_vp: *mut c_void, id: *mut c_char, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let id_str: String = CStr::from_ptr(id).to_str().unwrap().to_string();
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "input", 0
        );
        
        (*nn_ptr).add_input_neuron(id_str, neuron);
    }
}


/// Add hidden neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_hidden_neuron_ext(
    nn_vp: *mut c_void, id_len: usize, 
    max_edges: usize, neuron_level: u32
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        
        // Avoid duplicate ID's.
        let mut random_id: String = rand_id_gen(id_len);
        while (*nn_ptr).neural_net.hidden_neurons.contains_key(&random_id)
        {
            random_id = rand_id_gen(id_len);
        }

        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "hidden", neuron_level
        );
        
        // Indicate neuron level in ID.
        random_id += format!("[{}]", neuron_level).as_str();

        (*nn_ptr).add_hidden_neuron(random_id, neuron);
    }
}

/// Add output neuron.
#[unsafe(no_mangle)]
pub extern "C" fn add_output_neuron_ext(
    nn_vp: *mut c_void, id: *mut c_char, 
    max_edges: usize
)
{
    unsafe
    {
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let id_str: String = CStr::from_ptr(id).to_str().unwrap().to_string();
        let neuron: ArcNeuronTrait = create_neuron(
            max_edges, max_edges, 
            "output", MAX
        );
        
        (*nn_ptr).add_output_neuron(id_str, neuron);
    }
}

/// Conditionally select existing neuron from the neural network
/// to connect with a newly added one.
fn obtain_valid_neuron(
    target_neuron_id: &String,
    target_neuron_level: u32,
    neuron_group1: &HashMap<String, ArcNeuronTrait>,
    neuron_group2: &HashMap<String, ArcNeuronTrait>,
    connect_forward: bool,
    mut rand_gen: &mut rand::prelude::ThreadRng,
) -> String
{
    let selected_neuron_id: String;

    let (mut neuron_id_op, mut neuron_op);
    let mut neuron_id: String;

    loop
    {
        if rand_gen.gen_bool(0.5)
        {
            (neuron_id_op, neuron_op) = get_rand_neuron(
                neuron_group1, &mut rand_gen
            )
        }
        else
        {
            (neuron_id_op, neuron_op) = get_rand_neuron(
                neuron_group2, &mut rand_gen
            )
        }

        if neuron_op.is_some()
        {
            neuron_id = neuron_id_op.unwrap();
            
            if neuron_id != *target_neuron_id
            {
                let neuron: ArcNeuronTrait = neuron_op.unwrap();
                let neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
                let neuron_level: Option<u32> = neuron_guard.get_neuron_level();
                
                if neuron_level.is_some()
                {
                    let neuron_is_valid: bool;
                    let under_max_edges: bool;
                    
                    // Either checking to connect a forward or backward edge.
                    if connect_forward
                    {
                        // Next neuron must have higher level.
                        neuron_is_valid = target_neuron_level < neuron_guard.get_neuron_level().unwrap();
                        under_max_edges = neuron_guard.get_forward_edges().len() < neuron_guard.get_forward_edge_max();
                    }
                    else
                    {
                        // Previous neuron must have lower level.
                        neuron_is_valid = target_neuron_level > neuron_guard.get_neuron_level().unwrap();
                        under_max_edges = neuron_guard.get_backward_edges().len() < neuron_guard.get_backward_edge_max();
                    }

                    if neuron_is_valid && under_max_edges
                    {
                        selected_neuron_id = neuron_id;
                        break;
                    }
                }
                else
                {
                    // Neuron has no level, indicating its an input/output neuron.
                    selected_neuron_id = neuron_id;
                    break;
                }
            }
        }
    }

    return selected_neuron_id;
}