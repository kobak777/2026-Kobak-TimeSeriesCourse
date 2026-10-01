import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    if is_normalize:
        q_mean = np.mean(query)
        q_std = np.std(query)
        q_hat = (query - q_mean) / q_std if q_std > 0 else query - q_mean
    else:
        q_hat = query

    for i in range(N):
        subseq = ts[i: i + m]
        if is_normalize:
            sub_mean = np.mean(subseq)
            sub_std = np.std(subseq)
            sub_hat = (subseq - sub_mean) / sub_std if sub_std > 0 else subseq - sub_mean
        else:
            sub_hat = subseq

        dist_profile[i] = np.linalg.norm(q_hat - sub_hat)
            

    return dist_profile
