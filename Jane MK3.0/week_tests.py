#%%
# coding: utf-8
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


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

data = np.load("data/jan_09_2022/week.npy")
data.shape
#%%
train = data[:,:3,:]
validation = data[:,3:4,:]
test = data[:,4:,:]
train.shape,validation.shape,test.shape
#%%
#train = train[0]
#plt.subplots(1,2)
#plt.subplot(1,2,1)
#plt.xticks(np.arange(len(train)))
#plt.grid()
#plt.plot(train[0][0])
#plt.show()
train = train[0]
train_v = np.diff(train)# velocity / slope
#plt.subplot(1,2,2)
#plt.xticks(np.arange(len(train_v)))
#plt.plot(train_v[0][0])
prior_n = 6 # 30 minutes
priors = []
for day in train_v:
    p = []
    prior = [] # queue, with tail at 0 and head at n-1
    for i,price in enumerate(day):
        prior.append(price) # push to head

        if len(prior) > prior_n: #overflowing, remove
            prior.pop(0) # pop from tail

        if len(prior) == prior_n:
            p.append(np.mean(prior))

            # decision time
    priors.append(p)
#plt.plot(priors)
#plt.grid()
#plt.show()
v_avg = np.zeros((3,77-prior_n+1))

for i in range(prior_n,78):
    v_avg[:,i-prior_n] = np.mean(train_v[:,i-prior_n:i],axis=1)

# Testing this linalg way of computing these.
for i in range(3):
    assert (v_avg[i] == np.array(priors[i])).all()