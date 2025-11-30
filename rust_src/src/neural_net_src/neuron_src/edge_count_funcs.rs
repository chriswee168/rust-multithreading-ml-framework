use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Increment the output edge count.
pub fn increment_edge_count(edge_counter: &Arc<(Condvar, Mutex<(usize, usize)>)>)
{
    let mut edge_counts_guard: MutexGuard<'_, (usize, usize)> = edge_counter.1.lock().unwrap();
    edge_counts_guard.0 += 1;
}


/// If all edges has been visited, notify the main
/// thread to resume the neural network's forward method.
/// Only one neuron will notify.
pub fn edge_count_notify(edge_counter: &Arc<(Condvar, Mutex<(usize, usize)>)>)
{
    let edge_counts_guard: MutexGuard<'_, (usize, usize)> = edge_counter.1.lock().unwrap();
    if edge_counts_guard.0 == edge_counts_guard.1
    {
        edge_counter.0.notify_one();
    }
}