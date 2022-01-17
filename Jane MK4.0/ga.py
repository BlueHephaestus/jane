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
data = np.load(week_data_dir)

# Tensorflow models use float32
data = data.astype(np.float32)

# Function to roll parameters
def rollParams(uWs, top):
    '''This function takes in a list of unrolled weights (uWs) and a list with the number of neurons per layer in the following format:
    [input,first_hidden,second_hidden,output] and returns another list with the weights rolled ready to be input into a Keras model
    describing a two hidden layer neural network'''

    rWs = []
    s = 0

    for i in range(len(top) - 1):
        tWs = []
        for j in range(top[i]):
            tWs.append(uWs[s:s + top[i + 1]])
            s = s + top[i + 1]

        rWs.append(np.array(tWs))
        rWs.append(np.array(uWs[s:s + top[i + 1]]))
        s = s + top[i + 1]

    return rWs

# Fitness function
env = StaticStockEnvironment(data[0])

def stock_performance(agent):
    # Set up and get frozen model
    model.set_weights(rollParams(agent, [5, 10, 5, 1]))

    # Get all actions across all envdata
    outputs = model.predict_on_batch(env.env)

    # Execute all actions and get final reward
    reward = env.act(outputs)

    env.reset()

    return [reward]

# Create neural network model
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(units=1))

# Evolution settings

# Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox
toolbox = base.Toolbox()

# What we use to initialize models, drawing params from the uniform distribution
toolbox.register("attr_float", random.uniform, -1, 1)

# params for each individual, made up of 121 params drawn from uniform distribution
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 121)

# define our population as made up of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# define our fitness function
toolbox.register("evaluate", stock_performance)

# crossover, mutation, and selection operators for genetic algorithm
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


hof_n = 1
def train():
    pop = toolbox.population(n=100)

    # Get the best n individuals each generation, i.e. Hall of Fame
    # these are used to predict later when we finish
    hofs = tools.HallOfFame(hof_n)

    # initialize statistics to report
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Launch evolutionary algorithm
    _pop, _log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=10, stats=stats, halloffame=hofs, verbose=True)

    # Return best models (hall of fame)
    return hofs


cache = False
n = 20
offset = 20
# n = data.shape[0]
if not cache:
    training_rois = np.zeros((n,hof_n))
    validation_rois = np.zeros((n,hof_n))
    test_rois = np.zeros((n,hof_n))

    for i in tqdm(range(n), ncols=100):
        env = StaticStockEnvironment(data[i+offset])
        hofs = train()
        for hof_i,hof in enumerate(hofs):
            # Get performance for all models
            env.update_env(env.train)
            training_rois[i,hof_i] = stock_performance(hof)[0]
            env.update_env(env.validation)
            validation_rois[i,hof_i] = stock_performance(hof)[0]
            env.update_env(env.test)
            test_rois[i,hof_i] = stock_performance(hof)[0]

    np.save("jan_15_2022/training_roi.npy", training_rois)
    np.save("jan_15_2022/validation_roi.npy", validation_rois)
    np.save("jan_15_2022/test_roi.npy", test_rois)

else:
    training_rois = np.load("jan_15_2022/training_roi.npy")
    validation_rois = np.load("jan_15_2022/validation_roi.npy")
    test_rois = np.load("jan_15_2022/test_roi.npy")


def plotpts(data, label, i):
    plt.subplot(3,1,i)
    plt.plot(data)
    plt.title(label)
    for i, v in enumerate(data):
        plt.annotate("{:.2f}".format(v), xy=(i,v), xytext=(-7,7), textcoords='offset points')
    plt.legend()

for hof_i in range(hof_n):
    plt.subplots(3,1)
    training_roi = training_rois[:,hof_i]
    validation_roi = validation_rois[:,hof_i]
    test_roi = test_rois[:,hof_i]
    plotpts(training_roi, f"Training ROI = {np.sum(training_roi), np.mean(training_roi)}", 1)
    plotpts(validation_roi, f"Validation ROI = {np.sum(validation_roi), np.mean(validation_roi)}", 2)
    plotpts(test_roi, f"Test ROI = {np.sum(test_roi), np.mean(test_roi)}", 3)
    plt.show()
