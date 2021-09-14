import random
from collections import deque
from random import shuffle
import pandas as pd

import numpy as np

class Memory():
    def __init__(self, memory_size, batch_size):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    # Uniform sampling
    def sample(self):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size), size=self.batch_size, replace=False)
        batch = np.array([self.buffer[i] for i in index]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        return state, action, reward, state_prime, done

    # Uniform sampling the quantiles
    def sampleV2(self):
        f1, f2, f3, f4 = [], [], [], []
        mem_tra = np.array([self.buffer[i] for i in range(len(self.buffer))]).T.tolist()
        state = np.array(np.squeeze(mem_tra[0]), dtype=np.float32)
        df = pd.DataFrame(state, columns=['A', 'B', 'C', 'D'])

        try:
            df['bins'] = pd.cut(df['A'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            for pls in list(yo.index):
                f1.append(pls[1])
        except Exception as error:
            # print('ERRRRORRRRR: ', yo)
            f1 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['B'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            for pls in list(yo.index):
                f2.append(pls[1])
        except Exception as error:
            # print('ERRRRORRRRR: ', yo)
            f2 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['C'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            for pls in list(yo.index):
                f3.append(pls[1])
        except Exception as error:
            # print('ERRRRORRRRR: ', yo)
            f3 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['D'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            for pls in list(yo.index):
                f4.append(pls[1])
        except Exception as error:
            # print('ERRRRORRRRR: ', yo)
            f4 = random.sample(range(2000), 16)

        index = f1 + f2 + f3 + f4

        batch = np.array([self.buffer[i] for i in index]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        return state, action, reward, state_prime, done

    # Beta sampling with exploration in quantiles
    def sampleV3(self, epsilon):
        f1, f2, f3, f4 = [], [], [], []
        mem_tra = np.array([self.buffer[i] for i in range(len(self.buffer))]).T.tolist()
        state = np.array(np.squeeze(mem_tra[0]), dtype=np.float32)
        df = pd.DataFrame(state, columns=['A', 'B', 'C', 'D'])

        try:
            df['bins'] = pd.cut(df['A'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            if len(self.buffer) > 1000:
                if epsilon > 0.001:
                    epsilon *= 0.999
            for pls in list(yo.index):
                f1.append(pls[1])
        except Exception as error:
            print('Experience diversity problem in f1!')
            if epsilon < 0.1:
                epsilon *= 1.1
            f1 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['B'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            if len(self.buffer) > 1000:
                if epsilon > 0.001:
                    epsilon *= 0.999
            for pls in list(yo.index):
                f2.append(pls[1])
        except Exception as error:
            print('Experience diversity problem in f2!')
            if epsilon < 0.1:
                epsilon *= 1.1
            f2 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['C'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            if len(self.buffer) > 1000:
                if epsilon > 0.001:
                    epsilon *= 0.999
            for pls in list(yo.index):
                f3.append(pls[1])
        except Exception as error:
            print('Experience diversity problem in f3!')
            if epsilon < 0.1:
                epsilon *= 1.1
            f3 = random.sample(range(2000), 16)
        try:
            df['bins'] = pd.cut(df['D'], 16)
            yo = df.groupby('bins').apply(pd.DataFrame.sample)
            if len(self.buffer) > 1000:
                if epsilon > 0.001:
                    epsilon *= 0.999
            for pls in list(yo.index):
                f4.append(pls[1])
        except Exception as error:
            print('Experience diversity problem in f4!')
            if epsilon < 0.1:
                epsilon *= 1.1
            f4 = random.sample(range(2000), 16)

        index = f1 + f2 + f3 + f4

        batch = np.array([self.buffer[i] for i in index]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        return state, action, reward, state_prime, done, epsilon