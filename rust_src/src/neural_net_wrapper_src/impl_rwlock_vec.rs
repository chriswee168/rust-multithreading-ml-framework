use std::{iter::repeat, sync::{RwLockReadGuard, RwLockWriteGuard}};

use crate::neural_net_wrapper_src::neural_net_wrapper::NeuralNetWrapper;

impl NeuralNetWrapper
{
    /// Initialize the input gradient vector with zeros
    /// of specificed dimension.
    pub fn init_input_grad_vec(&self, in_dim: usize)
    {
        let vector: Vec<f32> = Vec::from_iter(repeat(0.0).take(in_dim));
        let mut write_guard: RwLockWriteGuard<'_, Vec<f32>> = self.input_rwlock_grad_vec.write().unwrap();
        *write_guard = vector;
    }

    /// Assign vector as input array to pass to the neural network for forward
    /// propagation.
    pub fn set_input_vec(&self, vector: Vec<f32>)
    {
        let mut write_guard: RwLockWriteGuard<'_, Vec<f32>> = self.input_rwlock_vec.write().unwrap();
        *write_guard = vector;
    }

    /// Obtain the read guard of input array gradients.
    pub fn get_input_grad_vec(&self) -> RwLockReadGuard<'_, Vec<f32>>
    {
        let read_guard: RwLockReadGuard<'_, Vec<f32>> = self.input_rwlock_grad_vec.read().unwrap();
        return read_guard;
    }

    
    /// Initialize the output vector with zeros
    /// of specificed dimension.
    pub fn init_output_vec(&self, out_dim: usize)
    {
        let vector: Vec<f32> = Vec::from_iter(repeat(0.0).take(out_dim));
        let mut write_guard: RwLockWriteGuard<'_, Vec<f32>> = self.output_rwlock_vec.write().unwrap();
        *write_guard = vector;
    }

    /// Assign vector as output gradient array to pass to the neural network for
    /// backpropagation.
    pub fn set_output_grad_vec(&self, vector: Vec<f32>)
    {
        let mut write_guard: RwLockWriteGuard<'_, Vec<f32>> = self.output_rwlock_grad_vec.write().unwrap();
        *write_guard = vector;
    }

    /// Obtain the read guard of the output array.
    pub fn get_output_vec(&self) -> RwLockReadGuard<'_, Vec<f32>>
    {
        let read_guard: RwLockReadGuard<'_, Vec<f32>> = self.output_rwlock_vec.read().unwrap();
        return read_guard;
    }
}