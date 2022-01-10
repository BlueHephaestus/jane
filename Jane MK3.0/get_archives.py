"""
Create archives for our datasets to run simulations and tests with.

This one just does raw numbers shaped according to timesteps. No symbols.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
from dateutil import parser
import pytz

import robin_stocks.robinhood as r
username, password = open("credentials.txt", "r").read().strip().split()
r.authentication.login(username=username, password=password)
symbols = []

def parsetime(t):
    return parser.parse(t["begins_at"]).astimezone(pytz.timezone("US/Eastern"))


def parsetimestr(t):
    # Return ticks, labels
    d = parsetime(t)
    return d.strftime("%Y-%b-%d(%a) @ %I:%M%p")

with open("data/nasdaq.txt") as f:
    f = f.readlines()

stock_n = len(f)

# format weekdata into shape (# days, # points) -> usually this is (5, 78)
# times obtained are 930am -> 355pm -> 78 points
weeks = np.zeros((stock_n, 5, 78))

# format 5 year data into shape (# points), this is maybe usually 1281? may be changed later.
# because 1281 days in 5 years, ish
fiveyears = np.zeros((stock_n, 1281))

for stock_i, line in enumerate(tqdm(f)):
    sym = line.strip().split("|")[0]

    # WEEK DATA #
    weekdata = r.stocks.get_stock_historicals(sym, interval="5minute", span="week", bounds="regular")

    day_offset = 0
    for i,timestep in enumerate(weekdata):
        day = parsetime(timestep).day
        if i == 0:
            # for initial day, so we can use it as indices
            day_offset = day
        day_i = day-day_offset
        min_i = i % 78
        weeks[stock_i,day_i,min_i] = timestep["close_price"]

    # 5 YEAR DATA #
    fiveyeardata = r.stocks.get_stock_historicals(sym, interval="day", span="5year", bounds="regular")

    for day_i,timestep in enumerate(fiveyeardata):
        fiveyears[stock_i,day_i] = timestep["close_price"]







    #times = [datetime.datetime.strptime(t["begins_at"], "%Y-%m-%dT%H:%M:%SZ").astimezone(pytz.timezone("US/Eastern")) for t in data]
    #timeticks = [parsetime(t) for t in data]
    #timelabels = [parsetimestr(t) for t in data]
    #prices = [float(t["close_price"]) for t in data]
    #print(len(timeticks))
    # plt.figure(figsize=(10,10))
    # plt.xticks(ticks=timeticks, labels=timelabels, rotation=45, ha="right")
    # plt.grid()
    #
    # plt.plot(timeticks, [float(t["close_price"]) for t in data])
    # plt.show()


np.save("data/jan_09_2022/week.npy", weeks)
np.save("data/jan_09_2022/5year.npy", fiveyears)
