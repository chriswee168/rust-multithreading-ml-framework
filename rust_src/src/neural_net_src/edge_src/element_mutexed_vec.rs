use std::{cell::RefCell, sync::Mutex};

// Vector where each element is allowed mutually exclusive access.
pub struct ElementMutexedVec<T>
{
    pub length: usize,
    pub data: RefCell<Vec<T>>,
    pub locks: Vec<Mutex<()>>
}
impl <T> ElementMutexedVec<T> {
    pub fn new(length: usize) -> Self
    {
        let data: RefCell<Vec<T>> = RefCell::new(Vec::with_capacity(length));
        let mut locks: Vec<Mutex<()>> = Vec::with_capacity(length);
        for _ in 0..length
        {
            locks.push(Mutex::new(()))
        }

        return Self
        {
            length,
            data,
            locks
        }
    }
}