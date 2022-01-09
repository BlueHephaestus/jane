#Constants
EPSILON = 1e-10

#HyperParameters
SIMULATION = True
VERBOSE = False
PERIOD_SIZE = 300 #(seconds) (same as historical)
INITIAL_STOCK_WALLET = 10 #Initial funds (USD) for a given stock

#Dynamically Determined

#Market. This helps ensure there is only one market ever during execution.
if SIMULATION:
  #SIMULATION
  from SimulatedMarket import SimulatedMarket
  MARKET = SimulatedMarket(PERIOD_SIZE)
else:
  #REAL LIFE
  from Market import Market
  MARKET = Market(PERIOD_SIZE)

STOCKS = MARKET.stocks
STOCK_N = len(STOCKS)
INITIAL_WALLET = {stock:INITIAL_STOCK_WALLET for stock in STOCKS}
