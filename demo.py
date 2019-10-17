import numpy as np
import ga
from matplotlib import pyplot as plt

func0 = lambda x, y:x + y
func1 = lambda x: np.sin(10 * x) * x + np.cos(2 * x) * x

func2 = lambda x, y: x * np.cos(2 * np.pi * y) + y * np.sin(2 * np.pi * x)

func4 = lambda x, y, z, h: x * np.sin(10 * z) + y * np.cos(2 * np.pi * x) * np.sin(5 * np.pi * h) - h * np.cos(y * h)


nums1 = [[np.random.rand() * 5] for _ in range(100)]
nums2 = list(zip(np.arange(-4, 4, 0.4), np.arange(-4, 4, 0.4)))
nums4 = list(zip(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5)))
bound1 = [(0, 5)]
bound2 = [(-4, 4), (-4, 4)]
bound4 = [(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
ga = ga.GA(nums4, bound4, func4, DNA_SIZE=10, cross_rate=0.8, mutation=0.005)
iter_time = 200
ga.plot_in_jupyter_1d(iter_time)

plt.plot(ga.best_fit)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.show()
