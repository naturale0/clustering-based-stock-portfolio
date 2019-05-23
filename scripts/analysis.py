from src.clustering import *
from src.portfolio import *
from functools import reduce
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle


def select_model(valid_res_return):
    valid_res_return = valid_res_return.loc[:, ["time", "kospi", "factors_8_GMV", "factors_8_Tangency"]]
    valid_res_return.columns = ["time", "kospi", "GMV", "Tangency"]
    return valid_res_return

def summarize_sd(selected):
    return selected.std()

def method_avg(x):
    n = x.shape[0]
    y = pd.DataFrame(x.loc[:, ["cumsum", "sd", "info_r"]] \
                      .to_numpy() \
                      .reshape(2, -1, order="F") \
                      .mean(axis=0) \
                      .reshape(n//2, -1, order="F") \
                      .round(3))
    y.columns = ["cumsum", "sd", "info_r"]

    return pd.concat([x.loc[:, ["with", "n_time"]] \
                       .drop_duplicates() \
                       .reset_index(drop=True), y], axis=1)

def cumret_plot(model):
    model = pd.melt(model, id_vars=['time'], value_vars=['kospi', "GMV", "Tangency"])
    model.columns = ["time", "asset", "logret"]
    model["asset"] = model.asset.astype("category")
    for a in model.asset.unique():
        model.loc[model.asset == a, "cumret"] = model.loc[model.asset == a, "logret"].cumsum().to_numpy()
        #model.loc[model.asset == a, "cumret_lin"] = np.exp(model.loc[model.asset == a, "cumret"].to_numpy())

    fig = plt.figure(figsize=(12,6))
    ax = sns.lineplot(x="time", y="cumret", hue="asset",
                      marker="+", mec=None, dashes=False, data=model)
    ax.xaxis.set_tick_params(rotation=90)

    plt.xlabel("Time", size=16)
    plt.ylabel("Cumulative return", size=16)
    plt.grid()
    plt.tight_layout()

    return fig, ax


with open("outputs/valid_res_list.pickle", "rb") as rb:
    valid_res = pickle.load(rb)
with open("outputs/test_res.pickle", "rb") as rb:
    test_res = pickle.load(rb)

kospi = pd.read_csv("data/processed/kospi.csv")


### Summary of `["result"]`
sd_summ = None
for i in range(len(valid_res)):
    newline = select_model(valid_res[i]["return"]).std()
    sd_summ = pd.concat([sd_summ, newline], axis=1) if sd_summ is not None else newline
sd_summ = sd_summ.T
sd_summ.to_csv("outputs/sd_summ2.csv", index=False)

summarize_sd(select_model(test_res["return"]))

### Summary of `["summary"]`
summ_list = np.dstack(map(lambda x: x["summary"].loc[:, ["cumsum", "sd", "info_r"]] \
                                                .to_numpy(), valid_res)).sum(axis=2)
summ_list = summ_list / len(valid_res)
summ_list = pd.DataFrame(summ_list, columns=["cumsum", "sd", "info_r"])

valid_summ = method_avg(pd.concat([valid_res[0]["summary"].loc[:, ["with", "n_time", "method"]].reset_index(drop=True), summ_list], axis=1))
valid_summ["rank"] = np.argsort(-valid_summ.info_r) + 1

test_summ = method_avg(test_res["summary"])
test_summ["rank"] = np.argsort(-test_summ.info_r) + 1


### Validation plot
model = select_model(pd.concat([v["return"] for v in valid_res]).reset_index(drop=True))
valid_cumret_total_plot = cumret_plot(model)
valid_cumret_part_plot = [cumret_plot(select_model(v["return"])) for v in valid_res]

valid_cumret_total_plot[0].savefig("outputs/valid_plot_00(2).png")
[fig.savefig(f"outputs/valid_plot_0{idx+1}(2).png") for idx, (fig, ax) in enumerate(valid_cumret_part_plot)]
print("figures saved")


### Test plot
test_ret = test_res["return"].loc[:, ["time", "kospi", "factors_8_GMV", "factors_8_Tangency"]]
test_ret.columns = ["time", "kospi", "GMV", "Tangency"]
test_ret.iloc[:, 1:] = test_ret.iloc[:, 1:].cumsum()

test_cumret_plot = cumret_plot(select_model(test_res["return"]))
test_cumret_plot[0].savefig("outputs/test-plot(2).png")
