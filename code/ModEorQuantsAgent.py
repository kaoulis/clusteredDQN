import csv

import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
import pandas as pd

from Memory import Memory


def network(input_shape, action_space, dense_layers, layer_neurons, optimizer):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(layer_neurons, input_shape=input_shape, activation='relu',
                                    kernel_initializer='he_uniform'))
    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(layer_neurons, activation='relu',
                                        kernel_initializer='he_uniform'))

    model.add(tf.keras.layers.Dense(action_space, activation='linear',
                                    kernel_initializer='he_uniform'))
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    return model


class DQNAgent:
    def __init__(self):
        # Environment
        self.env = gym.make('CartPole-v1')
        self.env.seed(7)
        self.rng = np.random.RandomState(7)
        self.env._max_episode_steps = 4000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000

        # Exploration-Exploitation
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.epsilon_max = 0.1
        self.train_start = 1000
        self.gamma = 0.95  # discount rate

        # Network
        self.optimizer = tf.keras.optimizers.Adam()
        # self.optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.dense_layers = 2
        self.layer_neurons = 64

        # Memory
        self.memory_size = 2000
        self.batch_size = 64
        self.memory = Memory(self.memory_size, self.batch_size)

        # Modifications (Quantiles sampling and Modified E-Greedy)
        self.modE = False
        self.quant = True

        # keep stats
        self.stats = {'e': [], 's': [], 'ep': [], 'avg': []}
        # create main model
        self.model = network((self.state_size,), self.action_size,
                             self.dense_layers, self.layer_neurons, self.optimizer)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # return random.randrange(self.action_size)
            return int(self.rng.randint(0, 2, 1).squeeze())
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory.buffer) < self.train_start:
            return
        if self.quant:
            state, action, reward, next_state, done = self.memory.sampleV2()
        else:
            state, action, reward, next_state, done = self.memory.sample()

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        start_taking_metrics = False
        total = 0
        e = 0
        while True:
            if self.epsilon <= self.epsilon_min + 0.01:
                start_taking_metrics = True
                e += 1
            if e == self.EPISODES:
                break
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                i += 1
                if start_taking_metrics:
                    total += 1

                # Modified-GREEDY
                if self.modE:
                    if len(self.memory.buffer) > self.train_start:
                        if not start_taking_metrics:
                            if self.epsilon > self.epsilon_min:
                                self.epsilon *= self.epsilon_decay
                        else:
                            if i > total / (e + 1):
                                if self.epsilon < self.epsilon_max:
                                    self.epsilon *= 1.1  # (1 + (1 - self.epsilon_decay))
                # E-Greedy
                else:
                    if len(self.memory.buffer) > self.train_start:
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                # REWARD SYSTEM
                if not done or i == self.env._max_episode_steps:
                    reward = reward
                else:
                    reward = 0

                self.memory.add((state, action, reward, next_state, done))
                state = next_state

                if done:
                    self.stats['e'].append(e)
                    self.stats['s'].append(i)
                    self.stats['ep'].append(self.epsilon)
                    self.stats['avg'].append(total / (e + 1))
                    print("episode: {}/{}, score: {}, e: {:.2}, avg: {}".format(e, self.EPISODES, i, self.epsilon,
                                                                                total / (e + 1)))
                    # MY-GREEDY AFTER DONE
                    if self.modE:
                        if start_taking_metrics:
                            self.epsilon = self.epsilon_min

                    if i == 4000:
                        continue
                self.replay()
        filename = 'Agent_' + str(self.dense_layers) + 'L_' + str(self.layer_neurons) + 'N.csv'
        df = pd.DataFrame({key: pd.Series(value) for key, value in self.stats.items()})
        df.to_csv(filename, encoding='utf-8', index=False)

    def test(self, name):
        self.load(name)
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

