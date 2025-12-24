use std::{collections::HashMap, fs::{File, OpenOptions}, io::Write, sync::MutexGuard};

use serde_json::{json, Value};

use crate::neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::core_deps::NeuronTrait, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}};

/// Obtain each neuron and write data to json file.
pub fn save_neurons(neuron_hashmap: &HashMap<String, ArcNeuronTrait>, json_path: String)
{
    let json_file: Result<File, std::io::Error> = File::create(json_path);
    
    if json_file.is_err()
    {
        println!("{}", json_file.err().unwrap())
    }

    else
    {
        let mut json_file: File = json_file.unwrap();
        let mut json_vec: Vec<Value> = Vec::new();
        for (_, neuron) in neuron_hashmap
        {
            let guard: MutexGuard<'_, Box<dyn NeuronTrait>> = neuron.lock().unwrap();
            // Obtain all necessary neuron info.
            let json_entry = json!({
                "backward_edge_max": guard.get_backward_edge_max(),
                "forward_edge_max": guard.get_forward_edge_max(),
                "neuron_id": guard.get_neuron_id(),
                "neuron_level": guard.get_neuron_level()
            });

            json_vec.push(json_entry);
        }

        // Write to JSON file.
        let json_str: String = serde_json::to_string(&json_vec).unwrap();
        let result: Result<(), std::io::Error> = json_file.write_all(json_str.as_bytes());
        if result.is_err()
        {
            println!("{}", result.err().unwrap())
        }
    }
}

/// Obtain each edge and write data to json file.
pub fn save_edges(edge_hashmap: &HashMap<String, ArcEdgeTrait>, json_path: String, edge_type: &str)
{
    let json_file: Result<File, std::io::Error> = File::create(json_path);
    
    if json_file.is_err()
    {
        println!("{}", json_file.err().unwrap())
    }

    else
    {
        let mut json_file: File = json_file.unwrap();
        let mut json_vec: Vec<Value> = Vec::new();
        for (_, edge) in edge_hashmap
        {
            let guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();

            // Obtain all necessary edge info.
            let json_entry: Value;
            let params: (f32, f32) = guard.get_params();

            if edge_type == "input"
            {
                let next_neuron: ArcNeuronTrait = guard.get_next_neuron().unwrap();
                let next_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = next_neuron.lock().unwrap();

                json_entry = json!({
                    "prev_idx": guard.get_prev_idx().unwrap(),
                    "next_neuron": next_neuron_guard.get_neuron_id(),
                    "pos_param": params.0,
                    "neg_param": params.1
                });
            }
            else if edge_type == "output"
            {
                let prev_neuron: ArcNeuronTrait = guard.get_prev_neuron().unwrap();
                let prev_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = prev_neuron.lock().unwrap();

                json_entry = json!({
                    "prev_neuron": prev_neuron_guard.get_neuron_id(),
                    "next_idx": guard.get_next_idx().unwrap(),
                    "pos_param": params.0,
                    "neg_param": params.1
                });
            }
            else
            {
                let prev_neuron: ArcNeuronTrait = guard.get_prev_neuron().unwrap();
                let next_neuron: ArcNeuronTrait = guard.get_next_neuron().unwrap();
                let prev_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = prev_neuron.lock().unwrap();
                let next_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = next_neuron.lock().unwrap();

                json_entry = json!({
                    "prev_neuron": prev_neuron_guard.get_neuron_id(),
                    "next_neuron": next_neuron_guard.get_neuron_id(),
                    "pos_param": params.0,
                    "neg_param": params.1
                });
            }

            json_vec.push(json_entry);
        }

        // Write to JSON file.
        let json_str: String = serde_json::to_string(&json_vec).unwrap();
        let result: Result<(), std::io::Error> = json_file.write_all(json_str.as_bytes());
        if result.is_err()
        {
            println!("{}", result.err().unwrap())
        }
    }
}