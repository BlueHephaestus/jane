#%%
# coding: utf-8

import numpy as np

"""
Remade environment for optimized training.

As opposed to LiveStockEnvironment, this is made without any real-time implementations, and meant to 
    be run for training, with many optimizations and functions implemented accordingly.
    
It will compute all the prices beforehand, for example, as well as return all the state values immediately,
    assuming a model that doesn't take reward into account until the end.
    
Because of this it also only takes in actions as a full vector, that it runs through and computes the final reward from.
"""


class StaticStockEnvironment:
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
        self.shares = 0 # Used for computing net worth at each timestep, and indicating if we are invested.
        self.wallet = self.INITIAL_WALLET # Used for computing net worth
        self.prices = self.get_prices() # Get all prices beforehand

    def get_prices(self):
        """
        Using the current state value - a feature vector - compute the current price per share of the stock,
            to be used in calculations for buying/selling and computing net worth.

        Since we only have the high,low,open, and close data of a stock at each timestep,
            we can't know exactly what the price will be at any given time in that timestep.

        So, we obtain the mean and variance of these values and use them to model a gaussian distribution,
            which we draw the price from.

        :return: sets the self.prices attribute as well as returns it
        """
        # generate all prices for env data as vector matching len of env
        means = np.mean(self.env[:,:-1], axis=1)
        stds = np.std(self.env[:,:-1], axis=1)
        self.prices = np.random.normal(loc=means, scale=stds)
        return self.prices

    def get_net_worth(self, price):
        """
        Using current simulation variables, return the current net worth of the bot.
        This is used as the reward at each timestep.

        Net worth = (shares * price/share) * (1 - self.TRADE_FEE) + self.wallet

        :param price: Price at timestep to compute net worth
        :return: net worth
        """
        return (self.shares * price) * (1.0 - self.TRADE_FEE) + self.wallet

    def update_env(self, data):
        """
        Reset this environment with new data, and recompute relevant data.
        """
        self.env = data
        self.shares = 0
        self.wallet = self.INITIAL_WALLET
        self.prices = self.get_prices()

    def reset(self):
        """
        Reset agent-specific attributes used to test this agent in this environment.

        DO NOT CHANGE THE PRICES
        :return:
        """
        self.shares = 0
        self.wallet = self.INITIAL_WALLET


    def act(self, actions):
        """
        Iterate through all actions and get final reward after all data.

        For each state value at timestep t:
            price is the actual buy price at that timestep
            action is the actual deicsion made by the agent at that timestep using that state

            by applying the action with the given price, we get our resulting reward.
            recall that the next state value is not determined by our actions in the stock market,
                since our own investments causing change would only occur with very very large investments.

            s_t+1, r_t+1 = environment(a_t*) # state does not depend on action, reward does.
            a_t = agent(s_t, r_t*) # agent does not use reward at the moment
            By combining them together we get the resulting reward for the next timestep

        :param actions: Vector of continuous values from an RL agent.
            action < -1 = sell (can only do this if shares > 0)
            action > 1 = buy (can only do this if shares == 0)
            -1 < action < 1 = do nothing

            For now it is an all-or-nothing, it can not invest portions of its net worth.

        :return:
            Iterates through all states, and computes final net worth
        """
        # Get all actions
        for action,price in zip(actions, self.prices):
            if action < -1:
                # SELL
                if self.shares > 0:
                    # Sell shares at current price and put it in the wallet.
                    self.wallet = (self.shares * price) * (1.0 - self.TRADE_FEE)
                    self.shares = 0

            elif action > 1:
                # BUY
                if self.shares == 0:
                    # Buy shares at current price, emptying our wallet.
                    self.shares = (self.wallet * (1.0 - self.TRADE_FEE)) / price
                    self.wallet = 0
            else:
                # DO NOTHING
                pass

        # Now everything is finalized, return final reward after the final action
        return self.get_net_worth(price=self.prices[-1])

