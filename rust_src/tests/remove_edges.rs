use std::u32::MAX;

use ai_core::{extern_funcs_src::remove_edge::remove_dead_ends, neural_net_src::{neuron_src::create_neuron::create_neuron, types_aliases::ArcNeuronTrait}, neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper};

#[test]
fn remove_edges()
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

    // Add input and output neurons.
    neural_net.add_input_edge("input", String::from("input_edge"), 0, 0.0, 0.0);
    neural_net.add_output_edge("output", String::from("output_edge"), 0, 0.0, 0.0);
    
    // Move neural net to a raw pointer.
    let neural_net: *mut NeuralNetWrapper = Box::into_raw(Box::new(neural_net));

    unsafe
    {
        // One input and output edge, plus 3 hidden edges connecting input, 
        // hidden and output neurons.
        assert_eq!(1, (*neural_net).neural_net.input_edges.len());
        assert_eq!(3, (*neural_net).neural_net.hidden_edges.len());
        assert_eq!(1, (*neural_net).neural_net.output_edges.len());

        // Remove the edge connecting the hidden neuron with the output neuron.
        remove_dead_ends(neural_net, String::from("edge1"), 1.0);
        
        // Edges going from input to hidden neuron should have been removed as it leads
        // to a dead end (as the hidden neuron has no forward/backward edge to propagate to).
        assert_eq!(1, (*neural_net).neural_net.input_edges.len());
        assert_eq!(1, (*neural_net).neural_net.hidden_edges.len());
        assert_eq!(1, (*neural_net).neural_net.output_edges.len());

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

        // Hidden neuron should not be connected to either the input or output neurons.
        assert_eq!(1, input_backward_edges.len());
        assert_eq!(1, input_forward_edges.len());
        assert_eq!(0, hidden_backward_edges.len());
        assert_eq!(0, hidden_forward_edges.len());
        assert_eq!(1, output_backward_edges.len());
        assert_eq!(1, output_forward_edges.len());
    }
}