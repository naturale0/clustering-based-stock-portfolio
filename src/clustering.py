import statsmodels.api as sm
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm


def time_slice(data, time_idx):
    as_list = list(time_idx)
    return stock_tbl.groupby(["code"]).nth(as_list).reset_index()


def time_expand(data, skip=[0,1]):
    cols = [col for col in range(data.shape[1]) if col not in skip]

    while len(set(data.time)) > 1:
        lagged = data.iloc[:, cols].shift(1)
        lagged.columns = [f"x{c}" for c in cols]

        data = pd.concat([data, lagged], axis=1)
        data = data.groupby(["code"], as_index=False).apply(lambda x: x.iloc[1:]).reset_index(drop=True)

        cols = np.array(cols) + len(cols)

    return data


def scale_tbl(data, skip=[0,1]):
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
      threshold: 주성분 개수 선택의 기준이 되는 변동의 비율 (0 ~ 1)

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
    idx_rf = [True if t in data.time.tolist() else False for t in risk_free.time]
    risk_free = risk_free.iloc[idx_rf, :]

    for code in tqdm(data.code.unique()):
        data.loc[data.code == code, "logret"] = data[data.code == code]["logret"].shift(-1)
        data.loc[data.code == code, "rf"] = risk_free.r.shift(-1)

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
    dfs = [data.loc[data.code == code, :].iloc[:-1, :] for code in tqdm(data.code.unique())]
    data = pd.concat(dfs)

    return data


def add_market_residual(data, market, risk_free):
    idx_mk = [True if t in data.time.tolist() else False for t in market.time]
    market = market.iloc[idx_mk, :]

    idx_rf = [True if t in data.time.tolist() else False for t in risk_free.time]
    risk_free = risk_free.iloc[idx_rf, :]

    for code in tqdm(data.code.unique()):
        data.loc[data.code == code, "mk"] = market.logret
        data.loc[data.code == code, "rf"] = risk_free.r
        y = data.logret - data.rf
        x = data.mk - data.rf

    lmfit = sm.OLS.from_formula("y ~ x", data=data).fit()
    yhat = lmfit.predict(lmfit)
    res = y - yhat
    data["market_res"] = (res - res.mean()) / res.std()

    return data


def kmeanspp(x, k, random_state=0):
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


def get_kmeans_tbl(data, ncmin=2, ncmax=5):
    data = data.dropna()

    ncs = range(ncmin, nvmax)

    X = data.iloc[:, 2:]
    models = [kmeanspp(X, nc) for nc in ncs]
    silhouettes = np.array([silhouette_score(X, m.labels_) for m in tqdm(models)])
    best_model = models[silhouettes.argmax()]

    return pd.DataFrame(np.array([data.code.unique(), kmeans.labels_]).T, columns=["code", "cluster"])


def kmeans_with(data, with_, market, risk_free):
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
    return np.log(np.sum(weight * np.exp(return_)))


def integrate_return_apply(row):
    return integrate_return(row["logret"], row["size"])


def get_cluster_return(data, time_idx, with_, market, risk_free):
    cluster_df = time_slice(data, time_idx)
    cluster_df = scale_tbl(cluster_df)
    cluster_df = kmeans_with(with_, market, risk_free)
    cluster_df = time_expand(cluster_df)
    cluster_df = get_kmeans_tbl(cluster_df)

    data = data.merge(cluster_df, how="left", on=["code"]).loc[:, ["code", "time", "logret", "size", "cluster"]]
    data["size"] = data.size.shift(1)
    data.dropna(inplace=True)

    data["logret"] = data.groupby(["cluster", "time"]).apply(integrate_return_apply).reset_index(drop=True)
    data = pd.DataFrame(pd.pivot_table(data, values='logret', index=['cluster']).T.to_records()).iloc[:, 1:]
    data.columns = [f"time{c}" for c in data.columns]
    data = data.T[0]

    x = data[time_idx]
    y = data[time_idx[-1] + 1]
    y_time = data.index[time_idx[-1] + 1]

    if (x.shape[0] == len(time_idx)) and (y_time == data.time.unique()[time_idx[-1]+1]):
        return {"x": x, "y": y}
    else:
        raise ValueError()
