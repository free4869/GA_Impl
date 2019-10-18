# encoding: utf-8
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
            raise Exception(f'范围与变量的数目不一致, {nums.shape[1]}个变量，{bound.shape[0]}个范围')

        for var in nums:
            for index, var_curr in enumerate(var):
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception(f'{var_curr}超出取值范围')

        for min_bound, max_bound in bound:
            if max_bound < min_bound:
                raise Exception(f'({min_bound}, {max_bound})非法的取值范围')

        # 所有变量的最小值和最大值
        # var_len为所有变量的取值范围大小
        # bit为每个变量按整数编码最小的二进制位数
        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        self.DNA_SIZE = DNA_SIZE
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.best_fit = []

        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        POP = np.zeros((*nums.shape, DNA_SIZE))
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                # 编码方式：
                num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
                # 用python自带的格式化转化为前面空0的二进制字符串，然后拆分成列表
                POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        self.POP = POP

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
    def get_fitness(self):
        result = self.func(*np.array(list(zip(*self.translateDNA()))))
        return result
# Calculate non-negative fitness based on min value
    def get_positive_fitness(self):
        result = self.func(*np.array(list(zip(*self.translateDNA()))))
        min_fit = np.min(result, axis=0)
        if not np.all(result == min_fit):
            result -= min_fit
        return result

    def select(self):
        fitness = self.get_positive_fitness()
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fitness / np.sum(fitness))]
# Crossover with high probability
    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, 2, size=(len(self.var_len), self.DNA_SIZE)).astype(np.bool)
                people[cross_points] = self.POP[i_, cross_points]
# Mutate with low probability
    def mutate(self):
        for people in self.POP:
            for var in people:
                for point in range(self.DNA_SIZE):
                    if np.random.rand() < self.mutation:
                        var[point] = 1 if var[point] == 0 else 1
# Evolution
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

# Print out current status
    def log(self):
        return pd.DataFrame(np.hstack((self.translateDNA(), self.get_fitness().reshape((len(self.POP), 1)))),
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
            self.best_fit.append(np.max(self.get_fitness()))
            print(self.log().sort_values(by="F", ascending=False).head(3))
