use std::ffi::{c_char, c_void, CStr};

use crate::{load_save_src::save::{save_edges, save_neurons}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

/// Save the parameters of neural network.
#[unsafe(no_mangle)]
pub extern "C" fn save_model_ext(nn_vp: *mut c_void, model_dir_str: *mut c_char)
{
    unsafe
    {
        let neural_net: *mut NeuralNetWrapper = nn_vp as *mut NeuralNetWrapper;
        let model_dir: String = CStr::from_ptr(model_dir_str).to_str().unwrap().to_string();
        // Save input, hidden and output neurons.
        save_neurons(&(*neural_net).neural_net.input_neurons, model_dir.clone() + "/input_neurons.json");
        save_neurons(&(*neural_net).neural_net.hidden_neurons, model_dir.clone() + "/hidden_neurons.json");
        save_neurons(&(*neural_net).neural_net.output_neurons, model_dir.clone() + "/output_neurons.json");

        // Save input, hidden and output edges.
        save_edges(&(*neural_net).neural_net.input_edges, model_dir.clone() + "/input_edges.json", "input");
        save_edges(&(*neural_net).neural_net.hidden_edges, model_dir.clone() + "/hidden_edges.json", "hidden");
        save_edges(&(*neural_net).neural_net.output_edges, model_dir.clone() + "/output_edges.json", "output");
    }
}