from src.preprocessing_utils import *
import os


def unpack_df(df):
    df = df.reset_index()
    df["time"] = df[['time', 'level_1']].astype(str).apply(lambda x: '-'.join(x), axis=1)
    df = pd.concat([df.time, df.price], axis=1)
    df.columns = ["time", "logret"]
    return df


path = os.path.join("data", "raw")
file_names = ["asset", "asset-growth", "equity", "equity-turnover",
              "leverage", "market-cap", "net-profit", "pcr", "per",
              "stock-number", "stock-price", "trade-amount", "volatility"]
var_names = ["asset", "asset_growth", "equity", "equity_turnover",
             "leverage", "market_cap", "net_profit", "pcr", "per",
             "stock_num", "price", "trade_amount", "volatility"]
extension = ".xls"


stock_tbl = preprocess(path, file_names, var_names, extension=".xls")

kospi = pd.read_excel("data/raw/kospi-index.xlsx", names=["time", "price"])
g = kospi.groupby([pd.DatetimeIndex(kospi.time).year, as_quarter(pd.DatetimeIndex(kospi.time).month)])
g = np.log(g.mean()).diff()
kospi = unpack_df(g)
kospi.head()

risk_free = pd.read_excel("data/raw/cd-risk-free.xlsx", names=["time", "r"])
risk_free.time = risk_free.time.str.replace("/", "-").str.split().str[0]
risk_free.r = np.log(1 + risk_free.r / 100)
risk_free.head()

try:
    os.mkdir("data/processed")
except:
    print("data/processed already exists")
    
#stock_tbl.to_csv("data/processed/stock_py.csv")
#kospi.to_csv("data/processed/kospi_py.csv")
#risk_free.to_csv("data/processed/risk_free_py.csv")