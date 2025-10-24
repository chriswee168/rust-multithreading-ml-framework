use rand::Rng;

/// Struct that defines default attributes for edges.
pub struct EdgeAttr
{
    // Parameters to use during forward propagation.
    // y = w * x + b
    // Weight values to use for negative and positive input.
    pub neg_weight: f32,
    pub pos_weight: f32,
    pub bias: f32, // Bias value to shift the product.
}

// Implement constructor method.
impl EdgeAttr
{
    pub fn new(weight_range: f32) -> Self
    {
        // Used for random weight initialization within specified range.
        let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
        return Self
        {
            neg_weight: rand_gen.gen_range(-weight_range..=weight_range),
            pos_weight: rand_gen.gen_range(-weight_range..=weight_range),
            bias: 0.0,
        }
    }
}

/// Contains trait methods for input, hidden and output edges.
pub trait EdgeTrait<T, U>
{
    fn forward(&self, input_val: f32) -> f32;
    fn backward(&mut self, input_val: f32, gradient_val: f32, lr: f32) -> f32;

    // Methods to obtain the previous/next neuron/index, typing depends on
    // edge struct.
    fn get_prev(&self) -> T;
    fn get_next(&self) -> U;
}