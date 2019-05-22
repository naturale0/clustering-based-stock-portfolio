from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import statsmodels.api as sm


def time_slice(data, time_idx):
    """
    select data entries with corresponding time_idx

    INPUT:
      data: DataFrame
      time_idx: time index in the form of "yyyy-q", where yyyy is a year and q is a quarter.

    RETURN:
      selected DataFrame
    """
    as_list = list(time_idx)
    return data.groupby(["code"]).nth(as_list).reset_index()


def time_expand(data, skip=[0,1]):
    """
    expand dataframe into wide form by time index

    INPUT:
      data: DataFrame
      skip: column indices to skip when expanding

    RETURN:
      expanded DataFrame
    """
    cols = [col for col in range(data.shape[1]) if col not in skip]

    while len(set(data.time)) > 1:
        lagged = data.iloc[:, cols].shift(1)
        lagged.columns = [f"x{c}" for c in cols]

        data = pd.concat([data, lagged], axis=1)
        data = data.groupby(["code"], as_index=False).apply(lambda x: x.iloc[1:]).reset_index(drop=True)

        cols = np.array(cols) + len(cols)

    return data


def scale_tbl(data, skip=[0,1]):
    """
    center and scale DataFrame

    INPUT:
      data: DataFrame
      skip: column indices to skip when scaling

    RETURN:
      scaled DataFrame
    """
    idx = [i for i in range(data.shape[1]) if i not in skip]
    vals = data.iloc[:, idx]
    data.iloc[:, idx] = (vals - vals.mean()) / vals.std()
    return data.reset_index(drop=True)


def pca_(data, skip=[0,1], threshold=.8):
    """
    데이터에 PCA(주성분분석)을 수행한다.

    INPUT:
      data: data frame
      skip: PCA 대상에서 제외할 열 번호 (integer vector)
      threshold: 주성분 개수 선택의 기준이 되는 변동의 설명 비율 (0 ~ 1)

    RETURN:
      변수들이 주성분으로 대체된 data frame
    """
    idx = [i for i in range(data.shape[1]) if i not in skip]
    omit_na = data.iloc[:, idx].dropna()    #### 여기 dropna() 때문에 데이터 절반이 날아감

    pca = PCA()
    x_pc = pd.DataFrame(pca.fit_transform(omit_na))
    n_pc = np.where(pca.explained_variance_ratio_.cumsum() > threshold)[0][0] + 1
    x_pc = x_pc.iloc[:, :n_pc]
    x_pc.columns = [f"PC{c+1}" for c in x_pc.columns]

    df_pc = pd.concat([data.dropna().iloc[:, skip].reset_index(drop=True), x_pc], axis=1)
    return df_pc


def add_factors_residual(data, risk_free):
    """
    add factors residual to the data

    INPUT:
      data: DataFrame
      risk_free: risk free DataFrame

    RETURN:
      concatenated DataFrame (by axis=1)
    """
    idx_rf = [True if t in data.time.tolist() else False for t in risk_free.time]
    risk_free = risk_free.iloc[idx_rf, :]

    for code in data.code.unique():
        data.loc[data.code == code, "logret"] = data[data.code == code]["logret"].shift(-1).to_numpy()
        data.loc[data.code == code, "rf"] = risk_free.r.shift(-1).to_numpy()

    y = data.logret - data.rf
    x = pca_(data.drop(['logret', 'rf'], axis=1))

    df_pca = pd.concat([x, y], axis=1).iloc[:, 2:]
    df_pca.rename(index=str, columns={0: "y"}, inplace=True)
    fml = " + ".join([c for c in df_pca.columns if "PC" in c])
    lmfit = sm.OLS.from_formula(f"y ~ {fml}", data=df_pca).fit()

    yhat = lmfit.predict(x)
    res = y - yhat
    data["factors_res"] = (res - res.mean()) / res.std()

    n_time = data.time.unique().shape[0]
    dfs = [data.loc[data.code == code, :].iloc[:-1, :] for code in data.code.unique()]
    data = pd.concat(dfs)

    return data


def add_market_residual(data, market, risk_free):
    """
    add factors residual to the data

    INPUT:
      data: DataFrame
      market: market index DataFrame (e.g. KOSPI)
      risk_free: risk free DataFrame

    RETURN:
      concatenated DataFrame (by axis=1)
    """
    idx_mk = [True if t in data.time.tolist() else False for t in market.time]
    market = market.iloc[idx_mk, :]

    idx_rf = [True if t in data.time.tolist() else False for t in risk_free.time]
    risk_free = risk_free.iloc[idx_rf, :]

    for code in data.code.unique():
        data.loc[data.code == code, "mk"] = market.logret.to_numpy()
        data.loc[data.code == code, "rf"] = risk_free.r.to_numpy()
    data["y"] = data.logret - data.rf
    data["x"] = data.mk - data.rf

    lmfit = sm.OLS.from_formula("y ~ x", data=data).fit()
    yhat = lmfit.predict(data["x"])
    res = data.y.to_numpy() - yhat
    data["market_res"] = (res - res.mean()) / res.std()

    return data


def kmeanspp(x, k, random_state=None):
    """
    fit K-means clustering with k groups

    INPUT:
      x: data to fit K-means algorithm
      k: number of groups
      random_state: random seed

    RETURN:
      fitted sklearn.KMeans() instance
    """
    #n = x.shape[0]
    #centers = [0] * k
    #centers[0] = np.random.randint(1, n+1)
    #
    #L2_mat = pd.DataFrame(squareform(pdist(x.iloc[:, 1:])), columns=x.index.unique(), index=x.index.unique())
    #L2_mat = L2_mat ** 2
    #
    #for i in range(1, k):
    #    weight = l2.iloc[:, centers].apply(np.min, axis=1)
    #    centers[i] = np.random.choice(range(1, n+1), p=weight/weight.sum())
    return KMeans(n_clusters=k, random_state=random_state).fit(x)


def get_kmeans_tbl(data, ncmin=2, ncmax=5, random_state=None):
    """
    find the best K-means fit and return clustering result as DataFrame

    INPUT:
      data: DataFrame to perform K-means clustering
      ncmin: minimum number of groups when performing K-means
      ncmax: maximum number of groups when performing K-means

    RETURN:
      clustering result as a DataFrame, with columns: ["code", "cluster"].
    """
    data = data.dropna()
    ncs = range(ncmin, ncmax+1)

    X = data.iloc[:, 2:]
    models = [kmeanspp(X, nc, random_state=random_state) for nc in ncs]
    silhouettes = np.array([silhouette_score(X, m.labels_) for m in models])
    best_model = models[silhouettes.argmax()]

    return pd.DataFrame(np.array([data.code.unique(), best_model.labels_]).T, columns=["code", "cluster"])


def kmeans_with(data, with_, market, risk_free):
    """
    concatenate data with additional variables, such as logret, market_residual, factors_res.
    this function does **not** performs K-means clustering. Rather, it prepares data shape for clustering.

    INPUT:
      data: DataFrame
      with_: additional variable to concatenate with 'data'
      market: market index DataFrame (e.g. KOSPI)
      risk_free: risk free DataFrame

    RETURN:
      DataFrame prepared for clustering
    """
    if with_ == "return":
        return data.loc[:, ["code", "time", "logret"]]
    elif with_ == "market_residual":
        return add_market_residual(data, market, risk_free).loc[:, ["code", "time", "market_res"]]
    elif with_ == "factors":
        return pca_(data.drop(["logret"], axis=1))
    elif with_ == "factors_residual":
        return add_factors_residual(data, risk_free).loc[:, ["code", "time", "factors_res"]]
    else:
        raise ValueError("'with_' should be one of ['return', 'market_residual', 'factors', 'factors_residual']")


def integrate_return(return_, weight):
    weight = np.array(weight)
    weight = weight / weight.sum()
    return np.log(np.sum(weight * np.exp(return_.tolist())))


def integrate_return_apply(row):
    """
    tweaked version of 'integrate_return' for use with pd.DataFrame().apply().
    """
    return integrate_return(row["logret"], row["size"])


def get_cluster_return(data, time_idx, with_, market, risk_free, random_state=None):
    cluster_df = time_slice(data, time_idx)
    cluster_df = scale_tbl(cluster_df)
    cluster_df = kmeans_with(cluster_df, with_, market, risk_free)
    cluster_df = time_expand(cluster_df)
    cluster_df = get_kmeans_tbl(cluster_df, random_state=random_state)

    data = data.merge(cluster_df, how="left", on=["code"]).loc[:, ["code", "time", "logret", "size", "cluster"]]
    data["size"] = data["size"].shift(1)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.cluster = data.cluster.astype(int)

    data = pd.DataFrame(data.groupby(["cluster", "time"]).apply(integrate_return_apply), columns=["logret"]).reset_index()
    pivot = pd.pivot_table(data, values='logret', index=['cluster'], columns=['time']).T
    data = pd.DataFrame(pivot.to_records()).iloc[:, 1:]
    data.columns = [f"cluster{int(c)+1}" for c in data.columns]
    data = pd.concat([pd.Series(pivot.index), data], axis=1)

    time_idx = time_idx.tolist()
    x = data.iloc[time_idx, :].reset_index(drop=True)
    y = data.iloc[time_idx[-1] + 1, :]
    y_time = data.time[time_idx[-1] + 1]

    if (x.shape[0] == len(time_idx)) and (y_time == data.time.unique()[time_idx[-1]+1]):
        return {"x": x, "y": y}
    else:
        print(time_idx)
        print(x.shape[0], len(time_idx))
        print(y_time, data.time.unique()[time_idx[-1]+1])
        raise ValueError()
