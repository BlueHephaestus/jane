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

# shape = (stock_n, 5, 78)

This will be a model that's working with each 78-vector.
So that it's making use of smaller minute to minute trends, rather than day to day.

We use first 3 days for training
#4 for validation
#5 for testing

Plan is this:

We have a model that uses the train data to pick optimal:
    prior_n (# of v used in running mean)
    v_threshold (slope)
    
    which it then uses on validation data for us to do testing
    
    it does this for every single stock independently, so that it has a suite of models.
        this can result in it producing a slope threshold that's huge, so it never does it.
    
    
"""

def get_velocity_prior(v, p, n):
    # Given velocity array v of shape ? x n
    # and prior size p
    # return velocity prior array.
    v_p = np.zeros((v.shape[0], n - p))  # prior velocities

    for i in range(p, n):
        v_p[:, i - p] = np.mean(v[:, i - p:i], axis=1)
    return v_p

def get_agent_roi(d, v_p, p, s):
    """
    :param d: Data vector from which velocity prior vector was obtained, matched to velocity prior
    :param v_p: Velocity prior vector (matches data points)
    :param p: Prior size
    :param s: Slope threshold
    :return: Total $ worth at end of given run assuming 100$ (or just %)
    """
    # We don't flatten v prior b/c we have to match it up to realtime in the sim
    # We'd only trade after p new values came in on the new day
    wallet = 100
    shares = 0
    invested = False


    for i,price in enumerate(d):

        # TODO try enforcing waiting p timesteps before selling
        if invested:
            # this will run immedately in the next timestep for now
            # cash out immediately
            # if we have 5 shares, and we sell at current price, we end up with 5 * whatever price it is selling for
            wallet = shares * price
            shares = 0
            invested = False

        # keep this as elif so we don't sell then immediately buy
        elif v_p[i] > s and not invested:
            # buy at price
            # if we have 10 dollars and it's 2$ a share, we have 5 shares now and no money
            shares = wallet / price
            wallet = 0
            invested = True

    # return worth at end of step - if invested, it's the shares, if not it's their wallet
    if invested:
        return shares * price
    else:
        return wallet

data = np.load("data/jan_09_2022/week.npy")
data.shape
#%%
train = data[:,:3,:]
validation = data[:,3:4,:]
test = data[:,4:,:]

stock_n = data.shape[0]
for stock_i in range(stock_n):
    n = train.shape[-1]  # day size

    # TODO test with acceleration, diff(n=2)
    v = np.diff(train[stock_i])# velocity / slope

    # GRID SEARCH FOR P AND S VALUES

    # prior size
    p_range = range(1,n-1)

    # slope threshold
    s_range = [
        1e-8, 5e-8,
        1e-7, 5e-7,
        1e-6, 5e-6,
        1e-5, 5e-5,
        1e-4, 5e-5,
        .001,.002,.003,.004,.005,.006,.007,.008,.009,
        .01,.02,.03,.04,.05,.06,.07,.08,.09,
        .1,.2,.3,.4,.5,.6,.7,.8,.9,
        1.0]

    results = np.zeros((len(p_range), len(s_range)))

    for i,p in enumerate(tqdm(p_range)):
        for j,s in enumerate(s_range):
            # shape 3 x n-p
            v_p = get_velocity_prior(v, p, n)

            # match our prices at the time the prior is computed to the prior values
            # so that if at time i we have a prior, we can be sure we're simulating the correct price
            d = train[stock_i,:,p:]

            # Now they both match in size, we flatten both to get vectors which our decision agent will run on
            v_p = v_p.flatten()
            d = d.flatten()

            assert len(v_p) == len(d)

            # Get results of slope thresholded decision agent on this
            roi = get_agent_roi(d, v_p, p, s)
            results[i,j] = roi

    #np.save("results.npy", results)
    best = np.argmax(results)
    i,j = best//len(s_range), best % len(s_range)
    print(f"\nBest ROI found at p={p_range[i]}, s={s_range[j]}, producing $ {results[i,j]}")


    #data = np.load("results.npy")
    #pd.DataFrame(data)
    plt.matshow(results)
    plt.show()

    p = p_range[i]
    s = s_range[j]
    #n = validation.shape[-1]
    v_p = get_velocity_prior(np.diff(validation[stock_i]), p, n).flatten()
    d = validation[stock_i,:,p:].flatten()
    assert len(v_p) == len(d)
    print(f"\tValidation ROI: $ {get_agent_roi(d, v_p, p, s)}")



