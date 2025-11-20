use std::sync::{Arc, MutexGuard, RwLock, RwLockWriteGuard};

use libai_core::{neural_net_src::{neuron_src::{core_deps::NeuronTrait, create_neuron::create_neuron, input_neuron}, rand_id_gen::rand_id_gen, types_aliases::{ArcNeuronTrait, NeuronBuffer}}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

#[test]
fn forward_pass()
{
    // Create an empty neural net.
    let mut neural_net: NeuralNetWrapper = NeuralNetWrapper::new();

    // Create a single input, hidden and output neuron, and add them to neural
    // net.
    let input_neuron: ArcNeuronTrait = create_neuron(
        5, 5, "input",
        neural_net.edge_counter.clone()
    );
    let hidden_neuron: ArcNeuronTrait = create_neuron(
        5, 5, "hidden",
        neural_net.edge_counter.clone()
    );
    let output_neuron: ArcNeuronTrait = create_neuron(
        5, 5, "output",
        neural_net.edge_counter.clone()
    );
    neural_net.add_input_neuron(String::from("input"), input_neuron.clone());
    neural_net.add_output_neuron(String::from("output"), output_neuron.clone());
    neural_net.add_hidden_neuron(String::from("hidden"), hidden_neuron.clone());

    neural_net.join_neurons("input", "hidden", String::from("edge1"), 0.9, -0.9);
    neural_net.join_neurons("hidden", "output", String::from("edge2"), -0.5, 0.9);

    let sample_input_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // Add input edges for each value in the sample vector.
    for i in 0..4
    {
        let random_id: String = rand_id_gen(10);
        neural_net.add_input_edge(
            "input", random_id, i, 
            1.0, 1.0
        );
    }

    // Add output edges to each output index, connecting from output neuron.
    for i in 0..4
    {
        let random_id: String = rand_id_gen(10);
        neural_net.add_output_edge(
            "output", random_id, i, 
            1.0, 1.0
        );
    }

    // Initialize input and output vectors.
    neural_net.init_input_vecs(4);
    neural_net.init_output_vecs(4);

    // Assign sample vector as input for neural net.
    neural_net.set_input_vec(sample_input_vec);

    // Create a neuron buffer to obtain neuron during forward pass simulation.
    let neuron_buffer: RwLock<NeuronBuffer> = RwLock::new(NeuronBuffer::new());
    let mut buffer_guard: RwLockWriteGuard<'_, NeuronBuffer> = neuron_buffer.write().unwrap();

    let mut input_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = input_neuron.lock().unwrap();

    // Dot product of input vector with input edges.
    // 1 * 1 + 2 * 1 + 3 * 1 + 4 * 1 = 10
    input_neuron_guard.forward(&mut buffer_guard);
    assert_eq!(10.0, input_neuron_guard.get_sum(true));

    // Forward propagation from input neuron to hidden neuron.
    // 10.0 * -0.9 = -9.0
    let mut hidden_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = hidden_neuron.lock().unwrap();
    assert_eq!(-9.0, hidden_neuron_guard.get_sum(true));

    // Hidden neuron has a visit count of 1 due to receiving a value from input neuron.
    assert_eq!(1, hidden_neuron_guard.get_visit_count(true));

    // Forward propagation from hidden neuron to output neuron.
    // -9.0 * -0.5 = 4.5
    hidden_neuron_guard.forward(&mut buffer_guard);

    // Hidden neuron visit count is zeroed as it propagates a value to output neuron.
    assert_eq!(0, hidden_neuron_guard.get_visit_count(true));
    
    let mut output_neuron_guard: MutexGuard<'_, Box<dyn NeuronTrait>> = output_neuron.lock().unwrap();
    assert_eq!(4.5, output_neuron_guard.get_sum(true));

    // Output neuron has a visit count of 1 due to receiving a value from hidden neuron.
    assert_eq!(1, output_neuron_guard.get_visit_count(true));

    // Output neuron propagates to output edges and thus output vector.
    // Validate output vector is correct.
    output_neuron_guard.forward(&mut buffer_guard);
    assert_eq!(vec![4.5, 4.5, 4.5, 4.5], *neural_net.output_rwlock_vec.read().unwrap());

    // Check if neuron buffer has copies of the neurons in the correct
    // order.
    assert!(Arc::ptr_eq(&buffer_guard[0], &hidden_neuron));
    assert!(Arc::ptr_eq(&buffer_guard[1], &output_neuron));
    assert_eq!(2, buffer_guard.len())
}