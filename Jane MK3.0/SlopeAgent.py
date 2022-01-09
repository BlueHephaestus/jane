from FastRunningMean import FastRunningMean
from config import *
import time

class SlopeAgent():
    
    def __init__(self, wallet, prior_size, slope_threshold):
        
        #Quantity of each stock owned, starts as 0
        self.inventory = {stock:0.0 for stock in STOCKS}
        
        #Money invested to obtain the inventory of each stock owned.
        self.investments = {stock:0.0 for stock in STOCKS}
      
        #Wallet (Amount of USD) given to each stock 
        self.wallet = wallet
        
        #Prior data to base decisions on
        self.prior_size = prior_size
        
        #Slope threshold to determine when to buy based on prior distribution.
        self.slope_threshold = slope_threshold
        
        #Slope data to use to base decisions on
        self.slopes = {stock:0.0 for stock in STOCKS}
        
        #Initialize Fast Running Mean Calculators for each stock's priors
        self.prior = {stock:FastRunningMean() for stock in STOCKS}
        
        #Current size of all priors.
        self.prior_n = 0

        #If this Agent is ready to execute actions
        self.active = False
        self.logged_active = False

        #Timing variables for profiling
        self.s = 0
        self.bs = 0
        self.ss = 0

    def update_prior(self, prices):
        #Update prior by adding these prices as new data
        self.prior_n += 1
        for stock in STOCKS:
            self.prior[stock].push(prices[stock])
            
        #If we've reached over capacity, we remove 
        #Remove/Deque head element of prior if over capacity
        #Prior additionally updates running mean and STD as a result
        if self.prior_n > self.prior_size:
            for stock in STOCKS:
                self.prior[stock].pop()
            self.prior_n -= 1

        #Set boolean indicating if prior is ready for analysis and agent is active
        #Agent is active when the prior has reached prior_size
        self.active = self.prior_n == self.prior_size
        
        #If it is active we update our prior distributions
        if self.active:
            
            if not self.logged_active:
                if VERBOSE:
                    print("Agent ACTIVE. Prior at Full Size {}/{}.".format(self.prior_n, self.prior_size))
                self.logged_active=True
            
            #Our Running Mean calculators are already up to date, we get the current means from them
            self.slopes = {stock:self.prior[stock].mean() for stock in STOCKS}

        else:
            if VERBOSE:
                print("Agent INACTIVE. Prior at Size {}/{}.".format(self.prior_n, self.prior_size))
    
    def buy(self, stock, prices):
        trade = MARKET.buy(stock, amount=self.wallet[stock])
        amount = float(trade["amount"]["amount"])
        
        #Use resulting info to update our investment variables for this stock
        self.inventory[stock] = amount
        
        #Market automatically applies fee to this when it computes how many shares to give us for our purchase
        #So we're good with this
        self.investments[stock] = self.wallet[stock]
        if VERBOSE:
            print("Spent {}$ for {} of {} @ {}/{}".format(self.investments[stock], self.inventory[stock], stock, prices[stock], stock))
        self.wallet[stock] = 0.0

    def sell(self, stock):
        #Make that trade boi
        trade = MARKET.sell(stock, amount=self.inventory[stock])
        unit_price = float(trade["unit_price"]["amount"])
        amount = float(trade["amount"]["amount"])
        total = float(trade["total"]["amount"])
        
        if VERBOSE:
            print("Sold {} {} @ {}/{}, Wallet: ${}".format(amount, stock, unit_price, stock, total))
        
        #Reset investment variables for this stock since we sold everything
        self.inventory[stock] = 0
        self.investments[stock] = 0
        
        #Update wallet with new returned total
        self.wallet[stock] = total
    
    def act(self, prices):
        #Given a new set of prices, decide whether to buy, sell, or do nothing, and execute action.

        #Action taken this turn.
        action = ""
        if self.active:
            #This agent needs the current price data in the prior to decide if we buy
            for stock in STOCKS:
                #We have money to spend
                if self.wallet[stock] > 0.0:
                    t = time.time()
                    #Our prior's slope is above the slope threshold in our data, take advantage of this and buy
                    if self.slopes[stock] > self.slope_threshold:
                        if VERBOSE:
                            print("{} @ {} with slope {} above buying threshold. Buying now.".format(stock, prices[stock], self.slopes[stock]))
                        self.buy(stock, prices)
                        action = "Buy"
                    self.bs+=(time.time()-t)
                    
                #We have stock to sell
                elif self.inventory[stock] > 0.0:
                    t = time.time()
                    #We immediately sell with this agent the timestep after we purchase
                    if VERBOSE:
                        print("{} @ {} :Selling now.".format(stock, prices[stock]))
                    self.sell(stock)
                    action = "Sell"
                    self.ss+=(time.time()-t)
                    
                if VERBOSE:
                    print("{} @ {}".format(stock, prices[stock]))

        #Update prior with current timestep's prices now that we've already taken action
        t = time.time()
        self.update_prior(prices)
        self.s+=(time.time()-t)
        
        #Return what action was taken if any
        return action
        
