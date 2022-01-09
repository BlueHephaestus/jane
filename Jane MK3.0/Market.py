from Robinhood import Robinhood
import numpy as np

class Market():
    def __init__(self, period_size):
        #self.period = 30 #seconds to wait until polling again
        self.period_size = period_size
        
        #Initialize API client
        self.client = Robinhood()
        username, password = open("credentials.txt","r").read().strip().split()
        self.client.login(username=username, password=password)
        
        #Load list of stocks
        self.stocks = np.load("data_stock_meta.npy")

        #Get instrument objects from RH API
        self.instruments = [self.client.instruments(stock)[0] for stock in self.stocks]
        self.prices = {stock:0.0 for coin in self.stocks}
        

    def buy(self, *args, **kwargs):
        #TODO
        return self.client.buy(*args, **kwargs)
        
    def sell(self, *args, **kwargs):
        #TODO
        return self.client.sell(*args, **kwargs)
    
    def poll_prices(self):
        #TODO
        #Poll API for all price data on our stocks
        return self.prices

    def __iter__(self):
        #Poll for prices every period seconds
        while True:
            yield self.poll_prices()
            time.sleep(self.period_size)

