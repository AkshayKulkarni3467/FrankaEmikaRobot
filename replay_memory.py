import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity,her_probability = 0.8):
        self.her_probability = her_probability
        self.capacity = capacity//2
        self.her_capacity = capacity//2
        self.buffer = []
        self.her_buffer = []
        self.position = 0
        self.her_position = 0

    def push(self, state, action, reward, next_state, done,her=False):
        if her:
            if len(self.her_buffer) < self.her_capacity:
                self.her_buffer.append(None)
            self.her_buffer[self.her_position] = (state, action, reward, next_state, done)
            self.her_position = (self.her_position + 1) % self.her_capacity
        else: 
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        her_batch_size = int(batch_size * self.her_probability)
        regular_batch_size = batch_size - her_batch_size
        batch = random.sample(self.buffer, regular_batch_size)
        her_batch = random.sample(self.her_buffer, her_batch_size)
        full_batch = list(batch + her_batch)
        state, action, reward, next_state, done = map(np.stack, zip(*full_batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)+ len(self.her_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
