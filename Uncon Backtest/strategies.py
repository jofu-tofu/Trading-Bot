import numpy as np
import pandas as pd
from numba import njit

def equal_weighting(signal):
    '''equal weighting of returns'''
    port = signal.div(signal.abs().sum(axis=1),axis=0)
    return port.fillna(0)
# --- Sparse table for rolling maximum ---
@njit
def build_sparse_table_max(arr):
    n = arr.shape[0]
    # Precompute logarithms
    log = np.empty(n + 1, dtype=np.int64)
    log[1] = 0
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1
    k = log[n] + 1  # number of levels
    st = np.empty((n, k), dtype=arr.dtype)
    for i in range(n):
        st[i, 0] = arr[i]
    j = 1
    while (1 << j) <= n:
        i = 0
        while i + (1 << j) - 1 < n:
            # Each entry is the maximum of two overlapping intervals
            left_val = st[i, j - 1]
            right_val = st[i + (1 << (j - 1)), j - 1]
            st[i, j] = left_val if left_val >= right_val else right_val
            i += 1
        j += 1
    return st, log

@njit(inline='always')
def query_max(st, log, L, R):
    j = log[R - L + 1]
    left_interval = st[L, j]
    right_interval = st[R - (1 << j) + 1, j]
    return left_interval if left_interval >= right_interval else right_interval

@njit
def dynamic_rolling_max_sparse(arr, win_sizes):
    n = arr.shape[0]
    res = np.empty(n, dtype=arr.dtype)
    st, log = build_sparse_table_max(arr)
    for i in range(n):
        w = int(win_sizes[i])
        L = i - w + 1
        if L < 0:
            L = 0
        res[i] = query_max(st, log, L, i)
    return res

# --- Sparse table for rolling minimum ---
@njit
def build_sparse_table_min(arr):
    n = arr.shape[0]
    log = np.empty(n + 1, dtype=np.int64)
    log[1] = 0
    for i in range(2, n + 1):
        log[i] = log[i // 2] + 1
    k = log[n] + 1
    st = np.empty((n, k), dtype=arr.dtype)
    for i in range(n):
        st[i, 0] = arr[i]
    j = 1
    while (1 << j) <= n:
        i = 0
        while i + (1 << j) - 1 < n:
            left_val = st[i, j - 1]
            right_val = st[i + (1 << (j - 1)), j - 1]
            st[i, j] = left_val if left_val <= right_val else right_val
            i += 1
        j += 1
    return st, log

@njit(inline='always')
def query_min(st, log, L, R):
    j = log[R - L + 1]
    left_interval = st[L, j]
    right_interval = st[R - (1 << j) + 1, j]
    return left_interval if left_interval <= right_interval else right_interval

@njit
def dynamic_rolling_min_sparse(arr, win_sizes):
    n = arr.shape[0]
    res = np.empty(n, dtype=arr.dtype)
    st, log = build_sparse_table_min(arr)
    for i in range(n):
        w = int(win_sizes[i])
        L = i - w + 1
        if L < 0:
            L = 0
        res[i] = query_min(st, log, L, i)
    return res

# --- Main business time signal function ---
def buy_high_sell_low_businesstime(buy_prices, sell_prices, log_trading_activity, enter_lookback=90000, exit_lookback=800):
    # Precompute dynamic window sizes as NumPy arrays.
    enter_lookback_np = np.ceil(enter_lookback / log_trading_activity).to_numpy()
    exit_lookback_np  = np.ceil(exit_lookback / log_trading_activity).to_numpy()
    
    # Convert prices to NumPy arrays.
    buy_np = buy_prices.to_numpy()
    sell_np = sell_prices.to_numpy()
    
    n_rows, n_cols = buy_np.shape
    dynamic_max_np = np.empty_like(buy_np)
    dynamic_min_np = np.empty_like(sell_np)
    
    # Loop over each column (you can further parallelize this loop if needed)
    for col in range(n_cols):
        dynamic_max_np[:, col] = dynamic_rolling_max_sparse(buy_np[:, col], enter_lookback_np[:, col])
        dynamic_min_np[:, col] = dynamic_rolling_min_sparse(sell_np[:, col], exit_lookback_np[:, col])
    
    # Reconstruct DataFrames.
    dynamic_max = pd.DataFrame(dynamic_max_np, index=buy_prices.index, columns=buy_prices.columns)
    dynamic_min = pd.DataFrame(dynamic_min_np, index=sell_prices.index, columns=sell_prices.columns)
    
    # Generate trading signals.
    enter_signal = (dynamic_max == buy_prices)
    exit_signal  = (dynamic_min == sell_prices)
    
    position = pd.DataFrame(
        np.where(exit_signal, 0, np.where(enter_signal, 1, np.nan)),
        index=buy_prices.index,
        columns=buy_prices.columns
    )
    position = position.ffill().fillna(0)
    
    return equal_weighting(position)  # assuming equal_weighting is defined elsewhere

def shock_reversal(returns, close_prices, entry_z_score_threshold = -1, exit_zscore_threshold = 1, entry_lookback = 15, exit_lookback = 15):
    '''shock reversal strategy'''
    z_scores = (returns - returns.rolling(window=entry_lookback, min_periods = 1).mean()).divide(returns.rolling(window=entry_lookback, min_periods = 1).std())
    entry_signal = (z_scores < entry_z_score_threshold).astype(int)

    close_zscores = (close_prices - close_prices.rolling(window=exit_lookback, min_periods = 1).mean()).divide(close_prices.rolling(window=exit_lookback, min_periods = 1).std())
    entry_signal2 = (close_zscores < 0).astype(int)
    exit_signal = (close_zscores > exit_zscore_threshold).astype(int)
    entry_signal2 = entry_signal2.where(entry_signal == 1, 0)
    position = pd.DataFrame(
        np.where(exit_signal, 0, np.where(entry_signal2, 1, np.nan)),
        index = returns.index,
        columns = returns.columns
    )
    position = position.ffill().fillna(0)
    return equal_weighting(position)