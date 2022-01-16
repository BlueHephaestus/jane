#%%
# coding: utf-8
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


"""
# This is extremely test-driven
# we don't have timestamps, symbols, etc.
# just testing if we can get some good predictions on 5-minute intervals.

# shape = (stock_n, 5, 78, 5)

This will be a model that's working with each 78-vector.
So that it's making use of smaller minute to minute trends, rather than day to day.

We use first 3 days for training
#4 for validation
#5 for testing

Plan is this:

Treat each timestep like a reinforcement learning bot,
    with each state input being the 5 values at that timestep
    the outputs being > 1 buy, < -1 sell, -1 < 0 < 1 do nothing
    the reward being the net worth at that timestep
    
    train a bot to maximize the net worth
"""


class StockEnvironment:
    def __init__(self, stock, initial_wallet=100, trade_fee=0.00):
        """
        A reinforcement learning environment, made from weekly stock data for ONE stock.
            to use all stock data, run this concurrently with others.

        More details in individual class functions.

        :param stock: Data for this stock. Will be in shape (# days, # timesteps / day, # features),
            e.g. shape of (5, 78, 5).
            This will by default only use the first 3 days as training data, and reserve the last two for
            validation and test data, respectively.
        :param initial_net_worth: Starting wallet for this environment for the agent to use.
            Used to determine the reward at each timestep, since this is one and the same.
        :param trade_fee: % taken from every trade, i.e. commission
        """
        # DATA
        self.train = stock[:3,:,:]
        self.validation = stock[3:4,:,:]
        self.test = stock[4:,:,:]
        self.feature_n = stock.shape[-1]

        # Additionally reshape each from (day #, timestep/day #, feature #) into (total timestep #, feature #)
        # so that we can iterate through multiple days in one fell swoop and simplify things for sims.
        # it also makes it a matrix and those are nice
        self.train = self.train.reshape(-1, self.feature_n)
        self.validation = self.validation.reshape(-1, self.feature_n)
        self.test = self.test.reshape(-1, self.feature_n)

        # CONSTANTS
        self.INITIAL_WALLET = initial_wallet
        self.TRADE_FEE = trade_fee

        # SIMULATION VARIABLES
        # Data currently being simulated, this is what will be referenced on each call to start() and act()
        # This can be changed when running validation and testing.
        self.env = self.train
        self.done = False
        self.t = 0 # TIMESTEP
        self.shares = 0 # Used for computing net worth at each timestep, and indicating if we are invested.
        self.price = 0 # current stock price
        self.wallet = self.INITIAL_WALLET # Used for computing net worth
        self.net_worth = self.wallet # initial net worth is always == wallet

    def get_net_worth(self):
        """
        Using current simulation variables, return the current net worth of the bot.
        This is used as the reward at each timestep.

        Net worth = (shares * price/share) * (1 - self.TRADE_FEE) + self.wallet

        :return: net worth
        """
        return (self.shares * self.price) * (1.0 - self.TRADE_FEE) + self.wallet

    def get_price(self, state):
        """
        Using the current state value - a feature vector - compute the current price per share of the stock,
            to be used in calculations for buying/selling and computing net worth.

        Since we only have the high,low,open, and close data of a stock at each timestep,
            we can't know exactly what the price will be at any given time in that timestep.

        So, we obtain the mean and variance of these values and use them to model a gaussian distribution,
            which we draw the price from.

        :return: Price value of current stock based on share price data.
        """
        mean,std = np.mean(state[:-1]), np.std(state[:-1])
        return np.random.normal(loc=mean,scale=std,size=1)

    def start(self):
        """
        Start simulation on current environment.
        If one is already running, this will reset it.
        :return: First state and reward values in format (state, reward)
            These will be the first timestep of data, and the initial wallet size, or net worth, of the agent.
        """
        self.done = False
        self.t = 0
        self.shares = 0
        self.wallet = self.INITIAL_WALLET
        self.state = self.env[self.t]
        self.price = self.get_price(self.state)
        self.net_worth = self.wallet # initial net worth is always == wallet
        return self.state, self.net_worth

    def act(self, action):
        """
        Execute action and get new state + reward value from training data.

        :param action: Continuous value from an RL agent.
            action < -1 = sell (can only do this if shares > 0)
            action > 1 = buy (can only do this if shares == 0)
            -1 < action < 1 = do nothing

            For now it is an all-or-nothing, it can not invest portions of its net worth.
        :return: Advances the environment one timestep forward, and returns (state, reward) tuple like the following:
            state: all price data at the next timestep
            reward: net worth at the next timestep, computed from state values
        """
        # ACTION PHASE - UPDATE SHARES AND WALLET
        if action < -1:
            # SELL
            if self.shares > 0:
                # Sell shares at current price and put it in the wallet.
                self.wallet = (self.shares * self.price) * (1.0 - self.TRADE_FEE)
                self.shares = 0

        elif action > 1:
            # BUY
            if self.shares == 0:
                # Buy shares at current price, emptying our wallet.
                self.shares = (self.wallet * (1.0 - self.TRADE_FEE)) / self.price
                self.wallet = 0
        else:
            # DO NOTHING
            pass

        self.t += 1
        self.done = self.t == self.env.shape[0]-1

        self.state = self.env[self.t]
        self.price = self.get_price(self.state) # recall, this is always used for backend, not shown to the agent
        self.net_worth = self.get_net_worth() # since at this point shares, price, and wallet are updated.

        return self.state, self.net_worth

