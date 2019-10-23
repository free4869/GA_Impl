# encoding=GBK
from matplotlib import pyplot as plt
import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from IPython import display

import numpy as np
import pandas as pd

class GA():

    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=1, mutation=0):
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        # Variable check
        if nums.shape[1] != bound.shape[0]:
            raise Exception(f'Unmatched number of variables and bounds, with {nums.shape[1]} variables and {bound.shape[0]} bounds')

        for var in nums:
            for index, var_curr in enumerate(var):
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception(f'{var_curr} exceeded the bound')

        for min_bound, max_bound in bound:
            if max_bound < min_bound:
                raise Exception(f'({min_bound}, {max_bound}) illegal bound')

        # min and max value of all nums
        # var_len is the range of nums
        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        self.DNA_SIZE = DNA_SIZE
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.best_fit = []

        # POP_SIZE stands for the number of population
        self.POP_SIZE = len(nums)
        POP = np.zeros((*nums.shape, DNA_SIZE))
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                # encoding method
                num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
                POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        self.POP = POP
        self.fitness = self.func(*np.array(list(zip(*self.translateDNA()))))

    def get_fitness(self):
        return self.fitness

# Decode
    def translateDNA(self):
        W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
        binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
        for i in range(binary_vector.shape[0]):
            for j in range(binary_vector.shape[1]):
                binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
                binary_vector[i, j] += self.bound[j][0]
        return binary_vector
# Calculate fitness
    def update_fitness(self):
        self.fitness = self.func(*np.array(list(zip(*self.translateDNA()))))

# Calculate non-negative fitness based on min value
    def get_positive_fitness(self):
        result = self.get_fitness()
        min_fit = np.min(result, axis=0)
        if not np.all(result == min_fit):
            result -= min_fit
        return result

    def select(self, with_rank = True, pin_percent = 0.1):
        pos_fitness = self.get_positive_fitness()
        sorted_fitness = np.argsort(pos_fitness)
        for i, v in enumerate(sorted_fitness):
            pos_fitness[v] = i

        POP_copy = self.POP.copy()
        for i in range(self.POP_SIZE):
            self.POP[i] = POP_copy[int(sorted_fitness[i])]

        pin_position = int(self.POP_SIZE * (1 - pin_percent))
        self.POP[0 : pin_position] = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=pin_position,
                                                               replace=True, p=pos_fitness / np.sum(pos_fitness))]

# Crossover with high probability
    def crossover(self, pin_percent = 0.1):
        pin_position = int(self.POP_SIZE * (1 - pin_percent))
        POP_copy = self.POP[0:pin_position]
        for people in POP_copy:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, 2, size=(len(self.var_len), self.DNA_SIZE)).astype(np.bool)
                people[cross_points] = self.POP[i_, cross_points]
# Mutate with low probability
    def mutate(self, pin_percent = 0.1):
        pin_position = int(self.POP_SIZE * (1 - pin_percent))
        POP_copy = self.POP[0:pin_position]
        for people in POP_copy:
            for var in people:
                for point in range(self.DNA_SIZE):
                    if np.random.rand() < self.mutation:
                        var[point] = 1 if var[point] == 0 else 1
# Evolution
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()
        self.update_fitness()
        self.best_fit.append(np.max(self.fitness))

# Print out current status
    def log(self):
        return pd.DataFrame(np.hstack((self.translateDNA(), self.fitness.reshape((len(self.POP), 1)))),
                            columns=[f'x{i}' for i in range(len(self.var_len))] + ['F'])

# Try drawing
    def plot_in_jupyter_1d(self, iter_time):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()

        for i in range(iter_time):
            '''
            plt.cla()
            x = np.linspace(*self.bound[0], self.var_len[0] * 50)
            plt.plot(x, self.func(x))
            x = self.translateDNA().reshape(self.POP_SIZE)
            plt.scatter(x, self.func(x), s=200, lw=0, c='red', alpha=0.5)
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
'''
            self.evolution()
            print("Iteration %d" % (i + 1))
            print(self.log().sort_values(by="F", ascending=False).head(3))
