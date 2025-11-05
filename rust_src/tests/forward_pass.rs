use std::sync::MutexGuard;

use libai_core::neural_net_src::{neural_net::NeuralNet, neuron_src::{core_deps::NeuronTrait, create_neuron::create_neuron, input_neuron}, types_aliases::ArcNeuronTrait};

#[test]
fn forward_pass()
{
    // Create an empty neural net.
    let mut neural_net: NeuralNet = NeuralNet::new(1);

    // Create a single input, hidden and output neuron, and add them to neural
    // net.
    let input_neuron: ArcNeuronTrait = create_neuron(
        5, 5, 0, "input"
    );
    let hidden_neuron: ArcNeuronTrait = create_neuron(
        5, 5, 1, "hidden"
    );
    let output_neuron: ArcNeuronTrait = create_neuron(
        5, 5, 1, "output"
    );
    neural_net.add_input_neuron(String::from("input"), input_neuron.clone());
    neural_net.add_output_neuron(String::from("output"), output_neuron.clone());
    neural_net.add_hidden_neuron(String::from("hidden"), hidden_neuron.clone());

    neural_net.join_neurons("input", "hidden", String::from("edge1"), 0.9, -0.9);
    neural_net.join_neurons("hidden", "output", String::from("edge2"), -0.5, 0.9);

    // Set input neuron value to 10.0.
    let test_input_value: f32 = 10.0;
    let mut input_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = input_neuron.lock().unwrap();
    input_neuron_guard.increment_sum(test_input_value);

    // Forward propagation from input neuron to hidden neuron.
    // 10.0 * -0.9 = -9.0
    input_neuron_guard.forward();
    let mut hidden_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = hidden_neuron.lock().unwrap();
    assert_eq!(-9.0, hidden_neuron_guard.get_sum());

    // Forward propagation from hidden neuron to output neuron.
    // -9.0 * -0.5 = 4.5
    hidden_neuron_guard.forward();
    let output_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = output_neuron.lock().unwrap();
    assert_eq!(4.5, output_neuron_guard.get_sum());
}