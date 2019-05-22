from quadprog import solve_qp
from itertools import product
from functools import reduce
from tqdm import tqdm
from .clustering import *
import pandas as pd
import numpy as np


def get_weight(x, method, risk_free=None):
    n_cluster = x.shape[1]
    zeros = np.zeros(n_cluster).T

    if method == "GMV":
        A = np.c_[np.ones(n_cluster), np.diag(np.ones(n_cluster))]
        b = np.array([1] + [0]*n_cluster, dtype=np.float64)
    elif method == "Tangency":
        if risk_free is None:
            raise ValueError("method is 'Tangency'. 'risk_free' should not be None")
        rf = risk_free.r.mean()
        A = np.c_[x.mean() - rf, np.diag([1]*n_cluster)]
        b = np.array([(x.mean() - rf).sum()] + [0]*n_cluster, dtype=np.float64)
    else:
        raise ValueError("invalid method")

    qp = solve_qp(x.cov().to_numpy(), zeros, A, b, meq=1)
    return qp[0]   # solution


def get_portfolio_return(data, timeset, n_time, with_, method,
                         market, risk_free, random_state=0):
    p_return = []
    for c_time in tqdm(timeset):
        time_idx = np.arange(c_time-6+1, c_time+1)
        c_return = get_cluster_return(data, time_idx, with_, market, risk_free)

        x = c_return["x"].iloc[:, 1:].reset_index(drop=True)
        y = c_return["y"][1:]

        weight = get_weight(x, method, risk_free.iloc[time_idx, :])
        p_return.append(integrate_return(y, weight))

    return pd.Series(p_return)


def expand_grid_py(*itrs):
    prod = list(product(*itrs))
    expanded = {'Var{}'.format(i+1): [x[i] for x in prod] for i in range(len(itrs))}
    return pd.DataFrame(expanded)


def expand_grid(*args):
    # all possible combinations of *args
    grd = expand_grid_py(*args)
    grd.sort_values(by=grd.columns.tolist(), inplace=True)
    return grd


def evaluate_portfolio(data, market, risk_free, start, end,
                       with_list, n_time_list, method_list, random_state=0):
    timeset = data.time.unique()
    st = [i for i, t in enumerate(timeset) if start in t][0]
    en = [i for i, t in enumerate(timeset) if end in t][0]
    timeset = list(range(st, en+1))

    # Model grid
    grd = expand_grid(with_list, n_time_list, method_list)
    grd.reset_index(drop=True, inplace=True)
    grd.columns = ["with", "n_time", "method"]

    model_names = reduce(lambda x, y: x.map(str) + "_" + y.map(str), [grd[c] for c in grd.columns])

    # Portfolio returns table
    pr_tbl = market.iloc[timeset, :]
    pr_tbl.columns = ["time", "kospi"]

    print(f"from {start} to {end}")

    for with_ in with_list:
        for n_time in n_time_list:
            for method in method_list:
                print(f"  with: {with_}")
                print(f"  n_time: {n_time}")
                print(f"  method: {method}")

                pr = get_portfolio_return(data, timeset, n_time, with_, method, market, risk_free)
                pr_tbl = pd.concat([pr_tbl, pr], axis=1)
                pr_tbl.columns = list(pr_tbl.columns[:-1]) + ["pr"]

    pr_tbl.columns = ["time", "kospi"] + model_names.tolist()

    # Model performance summary
    pr_cumsum = pr_tbl.iloc[:, 2:].cumsum().iloc[-1, :]
    pr_sd = np.diag(np.sqrt(pr_tbl.iloc[:, 2:].cov()))
    pr_info_rate = pr_tbl.iloc[:, 2:].to_numpy() - pr_tbl.iloc[:, 1].to_numpy()[np.newaxis].T
    pr_info_rate = pr_info_rate.mean(axis=0) / pr_info_rate.std(axis=0)

    summ = pd.concat([grd, pr_cumsum, pr_sd, pr_info_rate], axis=1)
    summ.columns = grd.columns.tolist() + ["cumsum", "sd", "info_r"]

    return {"return": pr_tbl, "summary": summ}
