"""
Improved from original static GA model to try a rolling model, that will predict at a given timestep t
    by taking in a window of data up to t-1.
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

class RollingEvolutionModel:
    """
    Model that evolves an ideal predictor for the next timestep based on a window of samples up to that timestep.
        Much like our "prior" method, but with more intense training before prediction.
    """
    def __init__(self, day_fname, stock_n=20, stock_offset=0, window_size=20):
        """

        :param day_fname: Filename of the day data to load. Will grab 5 days worth from this file.
        :param stock_n: Number of stocks to test on
        :param stock_offset: Offset before grabbing stock_n stocks. Used for paginated testing.
        :param window_size: Size of rolling window
        """
        # DATASET
        # Raw data of all stocks - currently shape (5436, 5, 78, 5)
        # Tensorflow models use float32
        self.data = np.load(f"{day_fname}/week.npy").astype(np.float32)

        # Parse down the data into our relevant sections
        self.data = self.data[stock_offset:stock_offset+stock_n]

        self.train = self.data[:, :3, :, :]
        self.validation = self.data[:, 3:4, :, :]
        self.test = self.data[:, 4:, :, :]

        # MODEL
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=5, activation='relu'))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(units=1))

        # EVOLUTION ALGORITHM
        # Creator
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Toolbox
        self.toolbox = base.Toolbox()

        # What we use to initialize models, drawing params from the uniform distribution
        self.toolbox.register("attr_float", random.uniform, -1, 1)

        # params for each individual, made up of 121 params drawn from uniform distribution
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, 121)

        # define our population as made up of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # define our fitness function - only per window
        self.toolbox.register("evaluate", self.window_fitness)

        # crossover, mutation, and selection operators for genetic algorithm
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # CONSTANTS
        self.stock_n = stock_n
        self.stock_offset = stock_offset
        self.window_size = window_size

        # DYNAMIC DATA - UPDATED AS WE RUN AND USED IN FUNCTIONS
        #self.stock_i = stock_offset
        #self.window_i = self.window_size  # beginning of current window, of size self.window_size
        #self.stock = self.data[self.stock_i, :, self.window_i:self.window_i+]

        # Handles splitting of train/val/test once we give it a singular stock
        # We just have to make sure to keep it limited to the window before our prediction.
        #self.env = StaticStockEnvironment(self.stock)

    def windows(self, stock_i, day_i):
        """
        Generator that iterates across windows for a given stock index and day index,
            as well as the timestep after each window.
            References self.data

        :param stock_i: Idx of stock
        :param day_i: Idx of day
        :return: yields windows and curr values until day data is exhausted
        """

        # Day shape will be (78, 5)
        day = self.data[stock_i, day_i]
        day_n = day.shape[0]

        # Window example for size 20: day[0:20], day[1:21], day[2:22], ..., day[57:77]
        # Curr example for size 20: day[20], day[21], day[22], ..., day[77]
        for i in range(day_n-self.window_size-1):
            window = day[i:i+self.window_size]
            curr = day[i+self.window_size]
            yield window, curr

    # Function to roll parameters
    @staticmethod
    def roll_params(uWs, top):
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

    def window_fitness(self, params):
        """
        Prepares a model using params, and returns its actions on the current environment data.
            Has to have this signature, so we reference a class variable for the data to get fitness on.
        :param params: Params of model
        :return:
        """
        # Set up model weights
        self.model.set_weights(self.roll_params(params, [5, 10, 5, 1]))

        # Get all actions across all
        outputs = self.model.predict_on_batch(self.window_env.env)

        # Execute all actions and get final reward (this also resets the env)
        reward = self.window_env.act(outputs)

        self.window_env.reset()

        return [reward]


    def train_on_window(self, window, shares, wallet):
        """
        Run a full evolutionary algorithm and return the best model, trained entirely on this window.

        :param window: Window of data from windows generator
        :return: One model, the best from training entirely on this window of data.
        """

        # MAKE SURE TO USE THE EXISTING WALLET AND SHARES
        self.window_env = StaticStockEnvironment(window, initial_shares=shares, initial_wallet=wallet)

        # Train a model on the current window, to predict on the current value.
        pop = self.toolbox.population(n=100)

        # Get the best n individuals each generation, i.e. Hall of Fame
        # these are used to predict later when we finish
        hofs = tools.HallOfFame(1)

        # initialize statistics to report
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Launch evolutionary algorithm
        _pop, _log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.8, mutpb=0.2, ngen=10, stats=stats, halloffame=hofs, verbose=False)

        # Return best model (hall of fame)
        return hofs[0]

    def evaluate_stock(self, stock_i):
        """
        This will go through the days for this stock, updating the shares and wallet via executing each action after a trained model
            for each window dictates what the next action will be.

        For each day:
            Go through the windows for this day, and for each window:
                train a window-specific model on the current environment - this means with the current shares and wallet
                get the model's action for curr - the value after the window.
                execute this action, updating the overall shares and wallet

        :param stock_i: Current stock idx
        :return: Final ROI after running on training days, validation days, and test days
        """
        # Env will be the actual days for this stock
        self.env = StaticStockEnvironment(self.data[stock_i]) # splits into train/val/test for us

        # self.env.train is shape (dayrange, 78, 5) fyi
        # unlike usual we will be calling act multiple times to fully simulate this range for the static env.
        roi = 0

        for day_i in range(5):
            for window_i,(window,curr) in enumerate(self.windows(stock_i, day_i)):

                if day_i == 3:
                    self.env.env = self.env.validation
                elif day_i == 4:
                    self.env.env = self.env.test
                # Obtain model for this window using our current shares and wallet
                curr_model = self.train_on_window(window, self.env.shares, self.env.wallet)

                # Use this model to act on curr, updating our shares and wallet

                # Set up model, execute action
                self.model.set_weights(self.roll_params(curr_model, [5, 10, 5, 1]))
                outputs = self.model.predict_on_batch(np.array([curr]))
                if day_i == 3 or day_i == 4:
                    i = window_i + self.window_size
                else:
                    i = day_i * self.data.shape[2] + window_i + self.window_size
                roi = self.env.act_single(outputs, i) # only execute this action on the curr_i timestep

                print(day_i, window_i, i, roi)
        return roi

rolling_model = RollingEvolutionModel("jan_12_2022", stock_n=20, stock_offset=0, window_size=70)
for i in range(1000):
    print(f"ROI for Stock {i}: {rolling_model.evaluate_stock(0)}")

