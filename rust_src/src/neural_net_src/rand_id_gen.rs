use rand::Rng;

/// Create a random string of characters of specified length.
pub fn rand_id_gen(id_len: usize) -> String
{
    let all_chars: Vec<char> = ('a'..='z').collect();
    let mut random_id: String = String::new();
    let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();
    for _ in 0..id_len
    {
        let random_index: usize = rand_gen.gen_range(0..all_chars.len());
        random_id += all_chars[random_index].to_string().as_str();
    }

    return random_id;
}