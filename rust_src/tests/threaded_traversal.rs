use core::time;
use std::{process::exit, time::Duration};

use libai_core::{neural_net_src::{neuron_src::create_neuron::create_neuron, rand_id_gen::rand_id_gen, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

#[test]
fn threaded_traversal()
{
    //////////////////////////////////////////////////////////
    // Create neural network with 4 input neurons, 2 hidden neurons
    // and 2 output neurons.
    // Neural network is organized as a regular feed forward dense network.
    let mut neural_net: NeuralNetWrapper = NeuralNetWrapper::new();
    
    // Create input neurons.
    for i in 0..4
    {
        let neuron = create_neuron(
            5, 5, "input",
        );
        let name: String = String::from("input") + i.to_string().as_str();
        neural_net.add_input_neuron(name, neuron);
    }
    // Create hidden neurons.
    for i in 0..2
    {
        let neuron = create_neuron(
            5, 5, "hidden",
        );
        let name: String = String::from("hidden") + i.to_string().as_str();
        neural_net.add_hidden_neuron(name, neuron);
    }
    // Create output neurons.
    for i in 0..2
    {
        let neuron = create_neuron(
            5, 5, "output",
        );
        let name: String = String::from("output") + i.to_string().as_str();
        neural_net.add_output_neuron(name, neuron);
    }

    let input_neuron_names = neural_net.neural_net.input_neurons.clone();
    let hidden_neuron_names = neural_net.neural_net.hidden_neurons.clone();
    let output_neuron_names = neural_net.neural_net.output_neurons.clone();
    
    // Join each input neuron with each hidden neuron.
    for (input_neuron_name, _) in &input_neuron_names
    {
        for (hidden_neuron_name, _)  in &hidden_neuron_names
        {
            neural_net.join_neurons(
                input_neuron_name, hidden_neuron_name, 
                rand_id_gen(10), 1.0, 1.0
            );
        }

        // Also join each input neuron with each output neuron.
        for (output_neuron_name, _)  in &output_neuron_names
        {
            neural_net.join_neurons(
                input_neuron_name, output_neuron_name, 
                rand_id_gen(10), 1.0, 1.0
            );
        }

        // Create input edges to obtain values from input vector.
        for i in 0..4
        {
            neural_net.add_input_edge(
                input_neuron_name, rand_id_gen(10), 
                i, 1.0, 1.0
            );
        }
    }

    // Connect each hidden neuron to each output neuron.
    for (hidden_neuron_name, _) in &hidden_neuron_names
    {
        for (output_neuron_name, _)  in &output_neuron_names
        {
            neural_net.join_neurons(
                hidden_neuron_name, output_neuron_name, 
                rand_id_gen(10), 1.0, 1.0
            );
        }
    }

    // Create input edges to send values to output vector.
    for (output_neuron_name, _)  in &output_neuron_names
    {
        for i in 0..4
        {
            neural_net.add_output_edge(
                output_neuron_name, rand_id_gen(10), 
                i, 1.0, 1.0
            );
        }
    }

    //////////////////////////////////////////////////////////
    
    // Spawn threads and initialize the input and output vectors.
    neural_net.spawn_threads(4, 0.0, true);
    neural_net.init_forward_buffer();
    neural_net.init_backward_buffer();

    // Check if output vector has the correct values after forward pass.
    let sample_input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..3
    {
        neural_net.init_input_vecs(4);
        neural_net.init_output_vecs(4);
        
        neural_net.set_input_vec(sample_input_vec.clone());
        neural_net.propagate();
        assert_eq!(vec![240.0, 240.0, 240.0, 240.0], *neural_net.get_output_vec());

        // Check each neuron has the correct values accumulated.
        for (_, neuron) in &neural_net.neural_net.input_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(10.0, guard.get_sum(true));
            assert_eq!(0, guard.get_visit_count(true));
        }

        for (_, neuron) in &neural_net.neural_net.hidden_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(40.0, guard.get_sum(true));
            assert_eq!(0, guard.get_visit_count(true));
        }

        for (_, neuron) in &neural_net.neural_net.output_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(120.0, guard.get_sum(true));
            assert_eq!(0, guard.get_visit_count(true));
        }

        // Test backpropagation and final input gradients.
        neural_net.set_output_grad_vec(vec![1.0, 1.0, 1.0, 1.0]);
        neural_net.prop_forward(false);
        neural_net.propagate();
        neural_net.prop_forward(true);
        assert_eq!(vec![96.0, 96.0, 96.0, 96.0], *neural_net.get_input_grad_vec());

        // Check each neuron has the correct gradient values accumulated.
        for (_, neuron) in &neural_net.neural_net.input_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(24.0, guard.get_sum(false));
            assert_eq!(0, guard.get_visit_count(false));
        }

        for (_, neuron) in &neural_net.neural_net.hidden_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(8.0, guard.get_sum(false));
            assert_eq!(0, guard.get_visit_count(false));
        }

        for (_, neuron) in &neural_net.neural_net.output_neurons
        {
            let guard = neuron.lock().unwrap();
            assert_eq!(4.0, guard.get_sum(false));
            assert_eq!(0, guard.get_visit_count(false));
        }
    }
}