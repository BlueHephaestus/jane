"""
Create archives for our datasets to run simulations and tests with.

This one just does raw numbers shaped according to timesteps. No symbols.

This one also grabs all the price data AND volume for every timestep, instead of just closing price
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

with open("nasdaq.txt") as f:
    f = f.readlines()

stock_n = len(f)

# format weekdata into shape (# days, # points per day, # features) -> usually this is (5, 78, 5)
# times obtained are 930am -> 355pm -> 78 points
weeks = np.zeros((stock_n, 5, 78, 5))

# format 5 year data into shape (# points), this is maybe usually 1281? may be changed later.
# because 1281 days in 5 years, ish
fiveyears = np.zeros((stock_n, 1281, 5))

for stock_i, line in enumerate(tqdm(f)):
    sym = line.strip().split("|")[0]

    # WEEK DATA #
    weekdata = r.stocks.get_stock_historicals(sym, interval="5minute", span="week", bounds="regular")

    # In the case of failed-to-retrieve, 404s, etc. we skip and leave as 0s
    if weekdata[0] == None: continue

    prev_day = -1
    day_i = -1
    for i,timestep in enumerate(weekdata):
        day = parsetime(timestep).day
        if prev_day != day:
            prev_day = day
            day_i += 1
        min_i = i % 78
        weeks[stock_i, day_i, min_i] = [timestep["high_price"], timestep["open_price"], timestep["close_price"], timestep["low_price"], timestep["volume"]]

    # 5 YEAR DATA #
    fiveyeardata = r.stocks.get_stock_historicals(sym, interval="day", span="5year", bounds="regular")

    for day_i,timestep in enumerate(fiveyeardata):
        fiveyears[stock_i,day_i] = timestep["high_price"], timestep["open_price"], timestep["close_price"], timestep["low_price"], timestep["volume"]



    # for day in range(5):
    #     plt.subplots(2,1)
    #     plt.subplot(2,1,1)
    #     plt.plot(weeks[stock_i,day,:,:4])
    #     plt.legend(["high", "open", "close" ,"low"])
    #     plt.subplot(2,1,2)
    #     plt.plot(weeks[stock_i,day,:,4])
    #     plt.legend(["volume"])
    #     plt.legend()
    #     plt.savefig(f"{day}.png")
    #     plt.show()
    #
    # break




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


print(weeks.shape)
print(fiveyears.shape)
np.save("jan_15_2022/week.npy", weeks)
np.save("jan_15_2022/5year.npy", fiveyears)
