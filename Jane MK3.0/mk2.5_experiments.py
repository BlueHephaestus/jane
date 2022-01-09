#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys, time

from config import *
from FastRunningMean import FastRunningMean
from SlopeAgent import SlopeAgent

def simulate_performance(prior_size, slope_threshold):
	#Initialize new Agent for this run
	agent = SlopeAgent(INITIAL_WALLET.copy(), prior_size, slope_threshold)
	
	#Buys and Sells the agent does
	actions = []

	#Timestep loop; try the agent with these parameters
	day_i = 0
	for i, prices in enumerate(MARKET):
		print(i)
		t1 = time.time()
		if prices == None:
			#End of a day
			
			#Sell any stocks we're still holding in our inventory
			for stock in agent.inventory:
				if agent.inventory[stock] > 0.0:
					agent.sell(stock)
					actions.append("Sell")
			if day_i == 2:
				#end of week, end simulation.
				break
			else:
				#New day
				day_i+=1
				
				#restart the agent from scratch with nothing changed but the money.
				#aka we make a new agent with the new earnings
				agent = SlopeAgent(agent.wallet, prior_size, slope_threshold)
				
				#then since this is only a marker timestep go to the next day without acting
				continue
				
		action = agent.act(prices)
		if action != "":
			#Agent bought or sold
			actions.append(action)
			
	#Get resulting number of trades executed by this agent via our actions array
	trades = len(actions)

	#Get total % profit obtained from our agent
	profits = {stock:(agent.wallet[stock]-INITIAL_WALLET[stock])/(max(INITIAL_WALLET[stock],EPSILON)) for stock in STOCKS}
	
	#Return simulated performance stats
	return trades, profits

results = {}
#trades, profits = simulate_performance(19, 5.12e-06)
trades, profits = simulate_performance(5, 5e-06)
#test performance with our parameters obtained via training set
profits = np.array(list(profits.values()))*100
plt.plot(np.arange(len(profits)), profits)
plt.show()
print(np.mean(profits))

# In[54]:

"""
for i,stock in enumerate(STOCKS):
	data = MARKET.data
	plt.plot(np.arange(len(data[0])), data[i])
	for j in range(len(data[0])-1,0,-len(data[0])//5):
		plt.axvline(x=j)
	plt.show()
	plt.title(stock)
	#print((data[-1]-data[0])/data[0])
"""
