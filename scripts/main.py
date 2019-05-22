from src.clustering import *
from src.portfolio import *
from functools import reduce
import pandas as pd
import numpy as np
import os
import pickle


# Random state
random_state = 0

# Load data
stock_tbl = pd.read_csv("data/processed/stock.csv")
kospi = pd.read_csv("data/processed/kospi.csv")
risk_free = pd.read_csv("data/processed/risk_free.csv")

# Models
with_list = ["return", "market_residual", "factors", "factors_residual"]
n_time_list = [6, 8, 10, 12]
method_list = ["GMV", "Tangency"]

# Validation period
start_list = ["2002-4", "2005-4", "2008-4", "2011-4"]
end_list = ["2005-3", "2008-3", "2011-3", "2014-3"]
valid_res = []
for st, en in zip(start_list, end_list):
    valid_res.append(evaluate_portfolio(stock_tbl, kospi, risk_free, st, en,
                                        with_list, n_time_list, method_list,
                                        random_state=random_state))

# Test period
start = "2014-4"
end = "2017-3"
test_res = evaluate_portfolio(stock_tbl, kospi, risk_free, start, end,
                              with_list, n_time_list, method_list)

# Save results
try:
    os.mkdir("outputs")
except OSError:
    print("outputs directory already exists")

with open("outputs/valid_res_list.pickle", "wb") as wb:
    pickle.dump(valid_res, wb)
with open("outputs/test_res.pickle", "wb") as wb:
    pickle.dump(test_res, wb)
