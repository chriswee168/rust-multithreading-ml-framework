"""
Python script for testing the performance of the adaptive Rust 
neural network.

This script uses the CartPole-v1 environment from the 
Gymnasium python library developed by the Farama Foundation.
Official link: https://gymnasium.farama.org/
Github link: https://github.com/Farama-Foundation/Gymnasium

"""
from python_src.config.NeuralNet import NeuralNet
from python_src.config.get_ctypes_funcs import get_ctypes_funcs
import numpy as np
import gymnasium as gym
import heapq
import platform
from copy import deepcopy
from python_src.HeapEntry import HeapEntry
import random

# Use path to shared library depending on operating system.
if platform.system() == "Windows":
    rust_lib_path = "rust_src/target/release/ai_core.dll"
elif platform.system() == "Linux":
    rust_lib_path = "rust_src/target/release/libai_core.so"

model_path = "python_src/models/cartpole-v1-model"
hyper_params = "python_src/hyper_params/cartpole-v1.json"
env_name = "CartPole-v1"
#render_mode = "human"
render_mode = None

env = gym.make(env_name, render_mode=render_mode)
rust_funcs = get_ctypes_funcs(rust_lib_path)

neural_net = NeuralNet(rust_funcs, hyper_params)
#neural_net.load(model_path)
neural_net.spawn_threads()
neural_net.display_params()

# Used to store sequences of experiences neural network is to be trained on.
heap_buffer: list[HeapEntry] = []
# Online buffer to record the observations and actions made by neural net.
obs_action_seq: list[tuple[np.ndarray, np.ndarray]] = []
selected_entry = None
reward_seq: list[float] = []
max_seq_len = 100
heap_buffer_max_size = 5000
heap_entry_idx = 0
io_arrays_idx = 0
ascending = True

stop_training = False
iterator = 0
loss_slope_total = 0
count = 0
max_epochs = 1000000
max_runs = 50000
mutation_interval = 100
    
def softmax(x: np.ndarray):
    """
    Create probability distribution of array x.
    
    :param x: Input numpy array.
    :type x: np.ndarray
    """
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def ce_loss(pred: np.ndarray, y: np.ndarray):
    """
    Categorical cross entropy loss.
    
    :param pred: Softmaxed array predicted by neural net.
    :param y: Target one hotted array.
    """
    return -np.sum(y * np.log(pred + 1e-8))

for i in range(max_epochs):
    total_loss = 0.0
    n_runs = 0
    
    obs_action_seq.clear()
    reward_seq.clear()

    total_reward = 0.0
    obs, _ = env.reset(seed=None)

    for j in range(max_runs):
        try:
            output = neural_net.forward(obs)
            softmax_probs = softmax(output)
            action = np.random.choice(2, 1, p=softmax_probs)[0]

            # Prepare entry to append to the online buffer.
            prev_obs = deepcopy(obs)
            output_arr = np.zeros(2, dtype=np.float32)
            output_arr[action] = 1
            entry = deepcopy((prev_obs, output_arr))

            # Perform action in environment and get reward.
            obs, reward, done, _, _ = env.step(action)
            if done:
                break

            # Continuous reward to tell how far away pole is from center.
            reward = 0.2 - abs(obs[0]) + 0.2 - abs(obs[2])

            # Add entry to online buffer and update reward sum.
            obs_action_seq.append(entry)
            reward_seq.append(reward)
            total_reward += reward
            if len(obs_action_seq) > max_seq_len:
                obs_action_seq.pop(0)
                total_reward -= reward_seq.pop(0)

            random_entry = deepcopy(random.choice(obs_action_seq))
            heapq.heappush(heap_buffer, HeapEntry(total_reward, random_entry))
            if len(heap_buffer) > heap_buffer_max_size:
                heapq.heappop(heap_buffer)
            
            # Every n passes, randomly add a neuron or join edges or
            # remove a random edge with parameters below a specified parameter
            # magnitude threshold.
            iterator += 1
            if iterator == mutation_interval:
                neural_net.expand()
                neural_net.reduce()
                iterator = 0
            
            if heap_buffer:
                selected_entry = heap_buffer[heap_entry_idx]
                input_array = selected_entry.entry[0]
                output_array = selected_entry.entry[1]

                # Train and optimize neural network.
                output = neural_net.forward(input_array)
                softmax_probs = softmax(output)

                total_loss += ce_loss(softmax_probs, output_array)

                # Backpropagate and perform gradient descent.
                grads = softmax_probs - output_array
                neural_net.optimize(grads)

                heap_entry_idx += 1

                # Signal for the next entry in the heap to be selected.
                if heap_entry_idx > len(heap_buffer) - 1:
                    heap_entry_idx = 0

            n_runs += 1

        except KeyboardInterrupt:
            stop_training = True
    
    print(f"{i}\t{len(heap_buffer)}\t{n_runs}\t{total_loss / n_runs}")
    if stop_training:
        break

# Save parameters.
neural_net.display_params()
neural_net.save(model_path)