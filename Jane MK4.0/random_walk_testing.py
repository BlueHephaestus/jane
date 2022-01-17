"""
Genetic Algo testing on stock environment sims.

Modified from https://github.com/cai91/openAI-classic-control/blob/master/cartPole_openAI.py

to work on our environments
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from LiveStockEnvironment import LiveStockEnvironment
from StaticStockEnvironment import StaticStockEnvironment


# General libraries
import random
import numpy as np
import matplotlib.pyplot as plt
#random.seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Genetic algorithms libraries
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

week_data_dir = "jan_15_2022/week.npy"
data = np.load(week_data_dir)[1000:2000]
plt.subplots(40,25)

for i,stock in enumerate(data):
    plt.subplot(40,25,i+1)
    # deltas across all timesteps
    prices = np.mean(stock[:,:,:-1], axis=2)

    # percent changes
    deltas = np.diff(prices)/prices[:,:-1]
    deltas = deltas.flatten()
    plt.hist(deltas, bins=20, range=(-.02,.02))

plt.show()
