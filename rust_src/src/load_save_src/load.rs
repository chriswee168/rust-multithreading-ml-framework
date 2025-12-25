use std::{ffi::{c_char, CString}, fs::read_to_string, os::raw::c_void};

use serde_json::Value;

use crate::{extern_funcs_src::{add_neurons::{add_input_neuron_ext, add_output_neuron_ext}, join_neurons::{add_input_edge_ext, add_output_edge_ext}}, neural_net_src::{neuron_src::create_neuron::create_neuron, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

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

///  Obtain the JSON entries to create the edges and use them to join the neurons.
pub fn load_edges(nn_vp: *mut c_void, json_path: String, edge_type: &str)
{
    let string_data: Result<String, std::io::Error> = read_to_string(json_path);    
    if string_data.is_err()
    {
        println!("{}", string_data.err().unwrap())
    }

    else
    {
        let json_entries: Vec<Value> = serde_json::from_str(&string_data.unwrap()).unwrap();
        let nn_ptr: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        for entry in json_entries
        {
            let pos_param: f32 = entry["pos_param"].as_f64().unwrap() as f32;
            let neg_param: f32 = entry["neg_param"].as_f64().unwrap() as f32;
            if edge_type == "input"
            {
                let neuron_id: &str = entry["next_neuron"].as_str().unwrap();
                let input_neuron_id_ptr: *mut c_char = CString::new(neuron_id).unwrap().into_raw();
                let prev_idx: usize = entry["prev_idx"].as_u64().unwrap() as usize;
                add_input_edge_ext(
                    nn_vp, input_neuron_id_ptr, prev_idx, 
                    neg_param, pos_param
                );
            }
            else if edge_type == "output"
            {
                let neuron_id: &str = entry["prev_neuron"].as_str().unwrap();
                let output_neuron_id_ptr: *mut c_char = CString::new(neuron_id).unwrap().into_raw();
                let next_idx: usize = entry["next_idx"].as_u64().unwrap() as usize;
                add_output_edge_ext(
                    nn_vp, output_neuron_id_ptr, next_idx, 
                    neg_param, pos_param
                );
            }
            else if edge_type == "hidden"
            {
                let prev_neuron_id: &str = entry["prev_neuron"].as_str().unwrap();
                let next_neuron_id: &str = entry["next_neuron"].as_str().unwrap();

                let edge_id: String = 
                    prev_neuron_id.to_string() + " --> " + next_neuron_id;
                
                unsafe
                {
                    (*nn_ptr).join_neurons(
                        prev_neuron_id, next_neuron_id, 
                        edge_id, neg_param, pos_param
                    );
                }
            }
        }
    }
}