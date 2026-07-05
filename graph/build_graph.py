import numpy as np

def build_A_norm(adj):
    A_hat = adj + np.eye(adj.shape[0])
    degree = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1 / np.sqrt(degree))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def build_static_graph(returns, threshold=0.5):
    corr = returns.corr()
    adj = (corr.abs() > threshold).astype(int)
    np.fill_diagonal(adj.values, 0)
    return build_A_norm(adj.values)

def build_weekly_snapshots(weekly_returns, tickers, corr_window=12, feat_window=12, threshold=0.5):
    A_norm_list, X_list, labels_list = [], [], []
    for t in range(corr_window, len(weekly_returns) - 1):
        window_data = weekly_returns.iloc[t-corr_window:t]
        corr_t = window_data.corr()
        adj_t = (corr_t.abs() > threshold).astype(int)
        np.fill_diagonal(adj_t.values, 0)
        A_norm_t = build_A_norm(adj_t.values)
        A_norm_list.append(A_norm_t)
        X_t = weekly_returns.iloc[t-feat_window:t].T.values
        X_list.append(X_t)
        label_t = (weekly_returns.iloc[t] > 0).astype(int).values
        labels_list.append(label_t)
    return A_norm_list, X_list, labels_list
