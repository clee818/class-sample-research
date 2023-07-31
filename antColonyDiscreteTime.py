# Swarm intelligence - Ant collony optimization
import random as rn
import numpy as np


class AntColony:
    def __init__(self, adjacency_mat, n_time_steps, n_ants, start, end,
                 alpha=0.1, beta=0.1, decay=0.1):
        # initializing all variables that we will use
        self.adjacency_mat = adjacency_mat  # adjacency matrix
        self.all_indices = np.arange(adjacency_mat.shape[0])
        self.pheromones = np.zeros(adjacency_mat.shape)  # creating matrix of pheromones
        self.start = start  # starting location of ant
        self.end = end  # ending point of path ant follows
        self.n_time_steps = n_time_steps  # number of iterations (aka waves)
        self.n_ants = n_ants  # number of ants
        self.best_path = []

        self.alpha = alpha  # weight of pheromone
        self.beta = beta  # weight of distance
        self.decay = decay  # decay rate of pheromone


    def run(self):
        paths = []
        for i in range(self.n_ants):
            path.append([]) # building path array
        for t in range(self.n_time_steps):
            self.run_time_step(paths, t, best_path)

    def run_time_step(self, paths, t, best_path):
        for a in range(self.n_ants):
            if paths[a][t] == self.end:
                self.p_add(paths[a][t])
                if (len(paths[t]) < len(self.best_path)) or len(
                        self.best_path) == 0: # checking if best path is found
                    self.best_path = paths[a][t]
            else:
                paths.append(self.gen_step(paths[a][t]))

    def gen_step(self, position):
        # pick ant's next move (random generation + pheromone probabilities)
        # return new position, otherwise (-1, -1) if no moves left

        pheromone = np.copy(self.pheromones)
        pheromone[self.adjacency_mat == 0] = 0

        dist = self.adjacency_mat[position[0]][position[1]]
        probs = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_probs = probs / probs.sum()
        position = np.random.choice(self.all_indices, 1, p=norm_probs)

    def p_add(self, path):
        for i in range(1, len(path)):
            self.pheromones[p - 1, p] = self.pheromones[p - 1, p] + (
                        10 / len(path))

    def p_decay(self):
        self.pheromones = self.decay * self.pheromones

# P(1/D)
# keeps track of pheromone strength and decay rate
