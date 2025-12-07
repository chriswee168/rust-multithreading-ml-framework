use std::{sync::{Arc, MutexGuard}, u32::MAX};

use libai_core::{neural_net_src::{edge_src::core_deps::EdgeTrait, neuron_src::create_neuron::create_neuron, types_aliases::{ArcEdgeTrait, ArcNeuronTrait}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

#[test]
fn join_neurons()
{
    // Create an empty neural net.
    let mut neural_net: NeuralNetWrapper = NeuralNetWrapper::new();

    // Create a single input, hidden and output neuron, and add them to neural
    // net.
    let input_neuron: ArcNeuronTrait = create_neuron(
        String::from("input"), 5, 5, "input", 0
    );
    let hidden_neuron: ArcNeuronTrait = create_neuron(
        String::from("hidden"), 5, 5, "hidden", 1
    );
    let output_neuron: ArcNeuronTrait = create_neuron(
        String::from("output"), 5, 5, "output", MAX
    );

    neural_net.add_input_neuron(String::from("input"), input_neuron.clone());
    neural_net.add_hidden_neuron(String::from("hidden"), hidden_neuron.clone());
    neural_net.add_output_neuron(String::from("output"), output_neuron.clone());

    // Join neurons to form a triangular topology.
    // input -> hidden, hidden -> output and input -> output.
    neural_net.join_neurons(
        "input", "hidden", String::from("edge"), 
        0.0, 0.0
    );
    neural_net.join_neurons(
        "hidden", "output", String::from("edge1"), 
        0.0, 0.0
    );
    neural_net.join_neurons(
        "input", "output", String::from("edge2"), 
        0.0, 0.0
    );

    let edge: &ArcEdgeTrait = neural_net.neural_net.hidden_edges.get("edge").unwrap();
    let edge1: &ArcEdgeTrait = neural_net.neural_net.hidden_edges.get("edge1").unwrap();
    let edge2: &ArcEdgeTrait = neural_net.neural_net.hidden_edges.get("edge2").unwrap();

    let edge_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge.lock().unwrap();
    let edge1_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge1.lock().unwrap();
    let edge2_guard: MutexGuard<'_, Box<dyn EdgeTrait>> = edge2.lock().unwrap();

    // Check edge connectivity.
    assert!(Arc::ptr_eq(&edge_guard.get_prev_neuron().unwrap(), &input_neuron));
    assert!(Arc::ptr_eq(&edge_guard.get_next_neuron().unwrap(), &hidden_neuron));
    
    assert!(Arc::ptr_eq(&edge1_guard.get_prev_neuron().unwrap(), &hidden_neuron));
    assert!(Arc::ptr_eq(&edge1_guard.get_next_neuron().unwrap(), &output_neuron));

    assert!(Arc::ptr_eq(&edge2_guard.get_prev_neuron().unwrap(), &input_neuron));
    assert!(Arc::ptr_eq(&edge2_guard.get_next_neuron().unwrap(), &output_neuron));

    // Get edges of each neuron.
    let input_neuron_guard = input_neuron.lock().unwrap();
    let output_neuron_guard = output_neuron.lock().unwrap();
    let hidden_neuron_guard = hidden_neuron.lock().unwrap();
    
    let input_backward_edges = input_neuron_guard.get_backward_edges();
    let input_forward_edges = input_neuron_guard.get_forward_edges();
    let hidden_backward_edges = hidden_neuron_guard.get_backward_edges();
    let hidden_forward_edges = hidden_neuron_guard.get_forward_edges();
    let output_backward_edges = output_neuron_guard.get_backward_edges();
    let output_forward_edges = output_neuron_guard.get_forward_edges();

    // Ensure neurons have the correct number of forward and backward edges.
    assert_eq!(0, input_backward_edges.len());
    assert_eq!(2, input_forward_edges.len());
    assert_eq!(1, hidden_backward_edges.len());
    assert_eq!(1, hidden_forward_edges.len());
    assert_eq!(2, output_backward_edges.len());
    assert_eq!(0, output_forward_edges.len());
    
}