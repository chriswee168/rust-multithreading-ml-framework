use std::{collections::HashMap, sync::{Arc, Condvar, Mutex, RwLockWriteGuard}};

use crate::neural_net_src::types_aliases::{ArcEdgeTrait, NeuronBuffer};

/// Struct that contains all attributes required for input,
/// hidden and output neurons.
pub struct NeuronAttr
{
    // Contains Arc references to edges indicating which neurons the current
    // neuron is connected to.
    pub forward_edges: HashMap<String, ArcEdgeTrait>,
    pub backward_edges: HashMap<String, ArcEdgeTrait>,

    pub max_backward_edges: usize,
    pub max_forward_edges: usize,

    forward_sum: f32, // Keep track of values during forward pass.
    backward_sum: f32, // Keep track of values during backward pass.

    // Used to control when the forward and backward sums are reset to zero
    // during the forward and backward pass.
    forward_visit_count: usize,
    backward_visit_count: usize,
}

impl NeuronAttr
{
    pub fn new(
        max_backward_edges: usize, max_forward_edges: usize, 
    ) -> Self
    {
        return Self 
        {
            forward_edges: HashMap::with_capacity(max_forward_edges), 
            backward_edges: HashMap::with_capacity(max_backward_edges), 
            max_backward_edges, max_forward_edges,
            forward_sum: 0.0, backward_sum: 0.0,
            forward_visit_count: 0, backward_visit_count: 0,
        }
    }

    /// Add a forward edge for this neuron to connect to another neuron.
    pub fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool
    {
        if !self.forward_edges.contains_key(&edge_id)
        {
            self.forward_edges.insert(edge_id, edge);
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Remove a forward edge to disconnect this neuron from another neuron.
    pub fn remove_forward_edge(&mut self, edge_id: &str) -> Option<ArcEdgeTrait>
    {
        let removed_edge: Option<ArcEdgeTrait> = self.forward_edges.remove(edge_id);
        return removed_edge;
    }

    /// Add a backward edge for this neuron to connect to a previous neuron.
    pub fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool
    {
        if !self.backward_edges.contains_key(&edge_id)
        {
            self.backward_edges.insert(edge_id, edge);
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Remove a backward edge to disconnect this neuron from a previous neuron.
    pub fn remove_backward_edge(&mut self, edge_id: &str) -> Option<ArcEdgeTrait>
    {
        let backward_edge: Option<ArcEdgeTrait> = self.backward_edges.remove(edge_id);
        return backward_edge;
    }

    /// Increment sum.
    pub fn add_to_sum(&mut self, value: f32, is_forward: bool)
    {
        if is_forward
        {
            self.forward_sum += value;
        }
        else 
        {
            self.backward_sum += value;
        }
    }

    /// Obtain the current sum of this neuron.
    pub fn get_sum(&self, is_forward: bool) -> f32
    {
        if is_forward
        {
            return self.forward_sum;
        }
        else 
        {
            return self.backward_sum;    
        }
    }

    /// Reset the sum of this neuron to zero.
    pub fn zero_sum(&mut self, is_forward: bool)
    {
        if is_forward
        {
            self.forward_sum = 0.0;
        }
        else 
        {
            self.backward_sum = 0.0;    
        }
    }

    /// Increment visit count.
    pub fn add_visit_count(&mut self, is_forward: bool)
    {
        if is_forward
        {
            self.forward_visit_count += 1;
        }
        else 
        {
            self.backward_visit_count += 1;
        }
    }

    /// Obtain the current visit count of this neuron.
    pub fn get_visit_count(&self, is_forward: bool) -> usize
    {
        if is_forward
        {
            return self.forward_visit_count;
        }
        else 
        {
            return self.backward_visit_count;    
        }
    }

    /// Reset the visit count of this neuron to zero.
    pub fn zero_visit_count(&mut self, is_forward: bool)
    {
        if is_forward
        {
            self.forward_visit_count = 0;
        }
        else 
        {
            self.backward_visit_count = 0;    
        }
    }
}

/// Contains trait methods for input, hidden and output neurons.
pub trait NeuronTrait: Send
{
    fn forward(&mut self, neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>); // Forward propagation.
    // Backward propagation.
    fn backward(
        &mut self,
        lr: f32, 
        return_grads: bool,
        neuron_buffer: &mut RwLockWriteGuard<'_, NeuronBuffer>
    );  // Backward propagation.

    // Wrapper methods for sum attributes in NeuronAttr.
    fn add_to_sum(&mut self, value: f32, is_forward: bool);
    fn get_sum(&self, is_forward: bool) -> f32;
    fn zero_sum(&mut self, is_forward: bool);

    // Wrapper methods for visit counter attributes in NeuronAttr.
    fn add_visit_count(&mut self, is_forward: bool);
    fn get_visit_count(&self, is_forward: bool) -> usize;
    fn zero_visit_count(&mut self, is_forward: bool);

    fn add_forward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool;
    fn remove_forward_edge(&mut self, edge_id: &str);
    fn add_backward_edge(&mut self, edge_id: String, edge: ArcEdgeTrait) -> bool;
    fn remove_backward_edge(&mut self, edge_id: &str);
    
    // Getter methods to access neuron edge connections.
    fn get_forward_edges(&self) -> &HashMap<String, ArcEdgeTrait>;
    fn get_backward_edges(&self) -> &HashMap<String, ArcEdgeTrait>;

    // Get neuron max edges.
    fn get_forward_edge_max(&self) -> usize;
    fn get_backward_edge_max(&self) -> usize;

    // For hidden neurons, return the level.
    fn get_neuron_level(&self) -> Option<u32> { None }
}