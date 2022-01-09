import numpy as np

class SimulatedMarket():
    def __init__(self, period_size):
        #Convert given period_size to match scale of historical data
        self.historical_period_size = 300
        self.period_size = int(period_size//self.historical_period_size)
        
        #self.data = np.load("data_stock_week.npy")[30:31,150:]
        #self.stocks = np.load("data_stock_meta.npy")[30:31]
        self.data = np.load("data_stock_week.npy")[:,150:]
        self.stocks = np.load("data_stock_meta.npy")
        self.prices = {stock:0.0 for stock in self.stocks}
        
        self.i = 0
        self.interval_i = 0
        
    @staticmethod
    def get_trade_fee(price):
        return 0
        
    def buy(self, stock, amount):
        #Amount is our $USD to spend on this account id.
        
        #Compute how much we will purchase after applying fee and then put that back
        #quantity = investment / price, investment = money spent - fee
        investment = amount - self.get_trade_fee(amount)
        quantity = investment / self.prices[stock]
        
        trade = {
            "amount":{"amount": str(quantity)}
        }
        return trade
        
    def sell(self, stock, amount):
        unit_price = self.prices[stock]
        subtotal = unit_price*amount
        fee = self.get_trade_fee(subtotal)
        total = subtotal-fee
        
        trade = {
                "amount":{"amount":str(amount)},
                "total":{"amount":str(total)},
                "unit_price":{"amount":str(unit_price)},
            }
        return trade
    
    def poll_prices(self):
        #Get next timestep of coin data.
        intervals = list(reversed([i for i in range(self.data.shape[1],0,-self.data.shape[1]//3)]))

        if self.i < intervals[self.interval_i]:
            for stock,price in zip(self.stocks, self.data[:,self.i]):
                self.prices[stock] = price
            self.i+=1
            return self.prices
        else:
            #Return None and reset our data, so that we know when the simulation ends
            #and can start a new simulation using the same SimulatedMarket instance.
            if self.interval_i == 2:
                #end of week reset
                self.i = 0
                self.interval_i = 0
            else:
                self.interval_i+=1
            return None
        
    def __iter__(self):
        while True:
            yield self.poll_prices()
