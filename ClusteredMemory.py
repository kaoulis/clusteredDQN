import math
import random
from collections import deque
import numpy as np


class CleverMemory():
    def __init__(self, memory_size, batch_size, clusters):
        self.buffer = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.clusters = clusters

    # Dynamic clustering while adding the experience
    def add(self, experience):
        # create clusters of size memory_size/ total_clusters until max number of clusters
        if len(self.buffer) < self.clusters:
            self.buffer.append(deque(maxlen=round(self.memory_size / self.clusters)))
            print('Cluster ', len(self.buffer), ' has been created!')
        for sub_buf in self.buffer:
            if not sub_buf:  # if empty
                sub_buf.append(experience)
                return
        # Pass the current experience in a np.array
        s = (np.squeeze(experience[0][0])).tolist()
        a = [experience[1]]
        r = [experience[2]]
        sp = (np.squeeze(experience[3][0])).tolist()
        done = [experience[4]]
        list1 = np.array(s + a + r + sp + done)

        # Pass the experiences of the center of each cluster in a np.array and measure the distance with the current
        # exp.
        best_sub_buffer_dist = 99999
        best_sub_buffer = 0
        i = 0
        for sub_buf in self.buffer:
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = (np.squeeze(mem[0][0][0])).tolist()
            a = [mem[0][1]]
            r = [mem[0][2]]
            sp = (np.squeeze(mem[0][3][0])).tolist()
            done = [mem[0][4]]
            list2 = np.array(s + a + r + sp + done)
            dist = np.linalg.norm(list1 - list2)
            if dist < best_sub_buffer_dist:
                best_sub_buffer = i
                best_sub_buffer_dist = dist
            i += 1

        self.buffer[best_sub_buffer].append(experience)

    # Uniform sampling each cluster.
    def sample(self):
        batches = deque(maxlen=self.batch_size)
        for sub_buf in self.buffer:
            batches += random.sample(sub_buf, int(self.batch_size / self.clusters))

        batch = np.array([batches[i] for i in range(len(batches))]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        return state, action, reward, state_prime, done

    # Normalised reversed weighted average sampling
    def weighted_sample(self, state):
        batches = deque(maxlen=self.batch_size)
        cur_state = np.squeeze(state)
        total_dist = 0
        total_weights = 0
        logs = []
        for sub_buf in self.buffer:  # get total distance
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = np.squeeze(mem[0][0][0])
            total_dist += np.linalg.norm(cur_state - s)

        for sub_buf in self.buffer:  # get total weights
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = np.squeeze(mem[0][0][0])
            total_weights += (1 - ((np.linalg.norm(cur_state - s)) / total_dist))

        for sub_buf in self.buffer:  # sample with inverse weights
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = np.squeeze(mem[0][0][0])
            weight = 1 - ((np.linalg.norm(cur_state - s)) / total_dist)
            inversed_weight = weight / total_weights
            logs.append([inversed_weight, int(inversed_weight * self.batch_size)])
            batches += random.sample(sub_buf, (int(inversed_weight * self.batch_size)))

        batch = np.array([batches[i] for i in range(len(batches))]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        # print(logs)
        return state, action, reward, state_prime, done

    # Reciprocal weighted average sampling
    def weighted_sampleV2(self, state):
        batches = deque(maxlen=self.batch_size)
        cur_state = np.squeeze(state)
        total_dist = 0
        logs = []
        for sub_buf in self.buffer:  # get total distance
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = np.squeeze(mem[0][0][0])
            total_dist += (1 / np.linalg.norm(cur_state - s))

        for sub_buf in self.buffer:  # sample with inverse weights
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            s = np.squeeze(mem[0][0][0])
            weight = ((1 / np.linalg.norm(cur_state - s)) / total_dist)
            logs.append([weight, int(weight * self.batch_size)])
            batches += random.sample(sub_buf, (int(weight * self.batch_size)))

        batch = np.array([batches[i] for i in range(len(batches))]).T.tolist()
        state = np.array(np.squeeze(batch[0]), dtype=np.float32)
        action = np.array(batch[1], dtype=np.int8)
        reward = np.array(batch[2], dtype=np.float32)
        state_prime = np.array(np.squeeze(batch[3]), dtype=np.float32)
        done = batch[4]

        # print(logs)
        return state, action, reward, state_prime, done

    # Beta dynamic clustering only for the state of the experience.
    def addV2(self, experience):
        if len(self.buffer) < self.clusters:
            self.buffer.append(deque(maxlen=round(self.memory_size / self.clusters)))
            print('Cluster ', len(self.buffer), ' has been created!')
        for sub_buf in self.buffer:
            if not sub_buf:  # if empty
                sub_buf.append(experience)
                return

        current_s = (np.squeeze(experience[0][0]))

        best_sub_buffer_dist = 99999
        best_sub_buffer = 0
        i = 0
        for sub_buf in self.buffer:
            mem = np.array([sub_buf[i] for i in range(len(sub_buf))]).tolist()
            cluster_s = (np.squeeze(mem[0][0][0]))

            dist = np.linalg.norm(current_s - cluster_s)
            if dist < best_sub_buffer_dist:
                best_sub_buffer = i
                best_sub_buffer_dist = dist
            i += 1

        self.buffer[best_sub_buffer].append(experience)
