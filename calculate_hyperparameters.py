import os.path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

import random
import time
from datetime import datetime
from collections import deque

print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class DQNAgent:
    def __init__(self, env, state_size, action_size, batch_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.001
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, actions_available, reward, next_state, done):
        self.memory.append((state, action, actions_available, reward, next_state, done))

    def one_hot_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def masked_predict(self, model, state, actions_available):
        act_values = model.predict(state, verbose=0)
        # mask other actions as they are not available to assign to vm
        for actions in actions_available:
            act_values[0][actions] = 0
        return act_values

    def act(self, state, actions_available):
        if np.random.rand() <= self.epsilon:
            return self.env.sample()
        act_values = self.masked_predict(self.model, state, actions_available)
        return np.argmax(act_values)

    def replay(self, episode):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        # Initialize arrays for inputs and targets
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, actions_available, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.masked_predict(self.target_model, next_state, actions_available)[0]))
            target_f = self.masked_predict(self.model, state, actions_available)
            # update reward for a given action
            target_f[0][action] = target

            # Store the states and the targets
            states[i] = state
            targets[i] = target_f

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + \
                           (1 - self.epsilon_min) * np.exp(-self.epsilon_decay_rate)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_epsilon(self):
        return self.epsilon

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class MarketEnvironment:
    def __init__(self, adj_matrix, vms):
        self.adj_matrix = adj_matrix
        self.vms = vms
        self.vm_status = np.zeros(len(vms))
        self.max_vm = max(vms)

    def reset(self):
        for key in self.adj_matrix:
            self.adj_matrix[key]["finished"] = False

        vm_status = [0] * len(self.vms)
        return vm_status

    def step(self, action):
        # Take an action in the environment and return the next state, reward, and whether the episode is done
        # action ->
        # 1. Which direction to we adjust the magnitude of the values of HP (Also might have to decide the value by which to adjust the value)
        # For ex : Delta + HP1, Delta - HP1
        # 2. calculate profit for given hyperparameters
        return next_state, reward, done

    def get_action_space_size(self):
        return len(self.adj_matrix)

    def get_state_space_size(self):
        return len(self.vms)

    def sample(self):
        unfinished_tasks = self.get_next_available_tasks()
        return -1 if len(unfinished_tasks) == 0 else random.choice(list(unfinished_tasks))



env = MarketEnvironment(adj_matrix, vms)

agent = DQNAgent(state_size=env.get_state_space_size(), action_size=env.get_action_space_size(), batch_size=64, env=env)

# Training the DQN agent
num_episodes = 10000
total_start = time.time()
weights_file_name = "weights.h5"
if os.path.exists(weights_file_name):
    agent.load(weights_file_name)
    print("Loaded existing weights...")
for episode in range(num_episodes):
    state = agent.one_hot_state(env.reset())
    reward_per_ep = 0
    for time_step in range(100):  # Adjust the maximum time steps as needed
        actions_available = env.get_next_available_tasks()
        action = agent.act(state, actions_available)
        next_state, reward, done, = env.step(action)
        # print(next_state, reward, done, info, _)
        reward_per_ep += reward

        next_state = agent.one_hot_state(next_state)
        agent.remember(state, action, actions_available, reward, next_state, done)
        agent.replay(episode)

        state = next_state

        if done:
            agent.target_train()
            break
    agent.save(weights_file_name)
    end_time = time.time()
    print(
        f"{datetime.utcnow()} : Episode {str(episode)} total reward:{str(reward_per_ep)} avg reward: {str(reward_per_ep / 500)} epsilon: {str(agent.get_epsilon())} time elapsed: {(end_time - total_start) / 60:.2f}min")
total_end = time.time()
print(f"Total time: {total_end - total_start}")