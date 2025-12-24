use std::{collections::HashMap, fs::{File, OpenOptions}, io::Write, sync::MutexGuard};

use serde_json::json;

use crate::{neural_net_src::{neuron_src::core_deps::NeuronTrait, types_aliases::ArcNeuronTrait}};

/// Obtain each neuron and write data to json file.
pub fn save_neurons(neuron_hashmap: &HashMap<String, ArcNeuronTrait>, json_path: String)
{
    let json_file: Result<File, std::io::Error> = 
        OpenOptions::new().create(true).append(true).open(json_path);
    
    if json_file.is_err()
    {
        println!("{}", json_file.err().unwrap())
    }

    else
    {
        let mut json_file: File = json_file.unwrap();
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
            
            // Write to JSON file.
            let json_str: String = serde_json::to_string(&json_entry).unwrap();
            let result: Result<(), std::io::Error> = json_file.write_all(json_str.as_bytes());
            if result.is_err()
            {
                println!("{}", result.err().unwrap())
            }
        }
    }
}