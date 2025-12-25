use std::{collections::HashMap, ffi::{c_char, CString}, fs::{read_to_string, File}, io::Write, os::raw::c_void, sync::MutexGuard};

use serde_json::{json, Value};

use crate::{extern_funcs_src::add_neurons::{add_input_neuron_ext, add_output_neuron_ext}, neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::{core_deps::NeuronTrait, create_neuron::create_neuron}, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Obtain the JSON entries to create the necessary neurons.
pub fn load_neurons(nn_vp: *mut c_void, json_path: String, neuron_type: &str)
{
    let string_data: Result<String, std::io::Error> = read_to_string(json_path);    
    if string_data.is_err()
    {
        println!("{}", string_data.err().unwrap())
    }

    else
    {
        let json_entries: Vec<Value> = serde_json::from_str(&string_data.unwrap()).unwrap();
        for entry in json_entries
        {
            let neuron_id: String = entry["neuron_id"].to_string();
            let neuron_id_char_ptr: *mut c_char = CString::new(neuron_id.clone()).unwrap().into_raw();
            let max_edges: usize = entry["backward_edge_max"].as_u64().unwrap() as usize;
            let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
            if neuron_type == "input"
            {
                add_input_neuron_ext(nn_vp, neuron_id_char_ptr, max_edges);
            }
            else if neuron_type == "output"
            {
                add_output_neuron_ext(nn_vp, neuron_id_char_ptr, max_edges);
            }
            else if neuron_type == "hidden"
            {
                let neuron_level: u32 = entry["neuron_level"].as_u64().unwrap() as u32;
                let neuron: ArcNeuronTrait = create_neuron(
                    neuron_id.clone(), max_edges, max_edges, 
                    "hidden", neuron_level
                );

                unsafe {(*nn_ptr).add_hidden_neuron(neuron_id, neuron)};
            }
        }
    }
}
