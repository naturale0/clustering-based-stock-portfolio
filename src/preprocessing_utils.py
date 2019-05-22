from functools import reduce
import pandas as pd
import numpy as np
import os


def as_quarter(months):
    """
    월(month) 값을 분기(quarter) 값으로 전환
    Example: (1, 2, 3, 4, 5, 6) -> (1, 1, 1, 2, 2, 2)

    INPUT:
      x: 1 ~ 12 사이의 integer vector
    RETURN:
      월 -> 분기로 전환된 integer vector
    """
    months = pd.to_numeric(months)
    if not np.all(np.isin(months, np.arange(1, 13))):
        raise ValueError("range of months exceeds 1~12")
    return (np.array(months) - 1) // 3 + 1


def is_quarter_interval(date_str):
    """
    시간 변수가 날짜 형식(ex: "20010131")인지
    혹은 분기 형식(ex: "2001/1 Quarter)인지 검사

    INPUT:
      x: date string vector
    RETURN:
      날짜 형식이면 FALSE, 분기 형식이면 TRUE
    """
    return len(str(date_str)) == 5


def reshape_long(df):
    """
    Short form 데이터를 long form 데이터로 변환
    시간 변수가 날짜 형식인 경우 분기 형식으로 변환

    INPUT:
      data: raw data를 불러들인 data frame
    RETURN:
      long form으로 전환된 data frame
    """
    df = gather(df)
    df.val = pd.to_numeric(df.val)
    df = tidyup_timeframe(df, is_quarter_interval(df.time[0]))
    return df


def gather(df):
    df.rename(index=str, columns={"Unnamed: 1": "code", "Unnamed: 2": "name"}, inplace=True)
    df.columns = ["y"+format_quarter(str(c)) if c[0].isnumeric() else c for c in df.columns]
    df = pd.wide_to_long(df, stubnames="y", i=["code", "name"], j="time")
    df = pd.DataFrame(df.to_records()).rename(index=str, columns={"y": "val"})
    return df


def format_quarter(string):
    return string.replace(" ", "").replace("/", "").replace("Quarter", "").replace("SemiAnnual", "2").replace("Annual", "4")


def tidyup_timeframe(df, is_quarter):
    df.val = pd.to_numeric(df.val)
    if is_quarter:
        year_quarter = df.time.astype(str).str.extract('(.{4,4})(.{1,1})')
        year_quarter.columns = ["year", "quarter"]
        yq = year_quarter.year + "-" + year_quarter.quarter
        yq.name = "time"
        df = pd.concat([df.iloc[:, :2], yq, df.val], axis=1)
    else:
        year_quarter = df.time.astype(str).str.extract('(.{4,4})(.{2,2})')
        year_quarter.columns = ["year", "quarter"]
        yq = year_quarter.year + "-" + as_quarter(year_quarter.quarter).astype(str)
        yq.name = "time"
        df = pd.concat([df.iloc[:, :2], yq, df.val], axis=1)
        df = pd.DataFrame(df.groupby(["code", "name", "time"]).mean().to_records())
    df.sort_values(by=["code", "time"], inplace=True)
    return df


def preprocess(path, file_names, var_names, extension=".xls"):
    dfs = []
    for name in file_names:
        print(name, end=", ")
        file_path = os.path.join(path, name+extension)
        data = reshape_long( pd.read_excel(file_path, skiprows=5).iloc[1:, 1:] )
        dfs.append(data)

    vals = reduce(lambda x, y: x.merge(y, how="left", on=["code", "name", "time"]), dfs).iloc[:, 3:]
    vals.columns = var_names

    features = extract_features(vals)
    df = pd.concat([data.loc[:, ["code", "time"]], features], axis=1)
    return df


def extract_features(vals):
    leverage = vals.leverage
    asset_growth = vals.asset_growth
    shares_turnover = vals.trade_amount / vals.stock_num
    roa = vals.net_profit / vals.asset
    roe = vals.net_profit / vals.equity
    size = vals.market_cap
    pcr = vals.pcr
    per = vals.per
    equity_turnover = vals.equity_turnover
    volatility = vals.volatility
    logret = np.log(vals.price).diff()

    features = pd.concat([leverage, asset_growth, shares_turnover, roa, roe, size,
                          pcr, per, equity_turnover, volatility, logret], axis=1)
    features.columns = ["leverage", "asset_growth", "shares_turnover", "roa", "roe", "size",
                        "pcr", "per", "equity_turnover", "volatility", "logret"]
    return features
