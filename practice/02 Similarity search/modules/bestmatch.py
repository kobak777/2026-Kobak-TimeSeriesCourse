import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone


    def perform(self):

        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using the naive algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of the naive algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf

        bestmatch = {
            'index' : [],
            'distance' : []
        }

        for i in range(N):
            subsequence = ts_data[i]

            if self.is_normalize:
                dist = DTW_distance(z_normalize(subsequence), z_normalize(query), self.r)
            else:
                dist = DTW_distance(subsequence, query, self.r)

            dist_profile[i] = dist
            if dist < bsf:
                bsf = dist

        topK_results = topK_match(dist_profile, excl_zone, self.topK)

        bestmatch['index'] = topK_results['indices']
        bestmatch['distances'] = topK_results['distances']

        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        lb_Kim = 0

        subs1 = np.asarray(subs1, dtype=float).ravel()
        subs2 = np.asarray(subs2, dtype=float).ravel()

        if len(subs1) != len(subs2):
            raise ValueError("Подпоследовательности должны иметь одинаковую длину.")

        if len(subs1) > 0:
            lb_Kim = (subs1[0] - subs2[0]) ** 2
            lb_Kim += (subs1[-1] - subs2[-1]) ** 2

        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        lb_Keogh = 0

        subs1 = np.asarray(subs1, dtype=float).ravel()
        subs2 = np.asarray(subs2, dtype=float).ravel()

        if len(subs1) != len(subs2):
            raise ValueError("Подпоследовательности должны иметь одинаковую длину.")

        radius = max(0, int(r))

        for i, value in enumerate(subs1):
            left = max(0, i - radius)
            right = min(len(subs2), i + radius + 1)
            lower = np.min(subs2[left:right])
            upper = np.max(subs2[left:right])

            if value > upper:
                lb_Keogh += (value - upper) ** 2
            elif value < lower:
                lb_Keogh += (value - lower) ** 2

        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf
        
        bestmatch = {
            'index' : [],
            'distance' : []
        }

        query = np.asarray(query, dtype=float).ravel()
        windows = np.asarray(ts_data, dtype=float)

        if self.is_normalize:
            std = np.std(query)
            query = (
                np.zeros_like(query, dtype=float)
                if std == 0
                else (query - np.mean(query)) / std
            )

            normalized_windows = []
            for window in windows:
                std = np.std(window)
                normalized_windows.append(
                    np.zeros_like(window, dtype=float)
                    if std == 0
                    else (window - np.mean(window)) / std
                )
            windows = np.asarray(normalized_windows)

        radius = min(int(self.r * m), m - 1)
        active = np.ones(N, dtype=bool)

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

        for _ in range(min(max(int(self.topK), 0), N)):
            bsf = np.inf
            dist_profile[:] = np.inf

            for i in range(N):
                if not active[i]:
                    continue

                candidate = windows[i]

                lb = self._LB_Kim(query, candidate)
                if lb > bsf:
                    self.lb_Kim_num += 1
                    continue

                # QC: запрос относительно envelope кандидата.
                lb = self._LB_Keogh(query, candidate, radius)
                if lb > bsf:
                    self.lb_KeoghQC_num += 1
                    continue

                # CQ: кандидат относительно envelope запроса.
                lb = self._LB_Keogh(candidate, query, radius)
                if lb > bsf:
                    self.lb_KeoghCQ_num += 1
                    continue

                self.not_pruned_num += 1

                # DTW с окном Sakoe–Chiba.
                previous = np.full(m + 1, np.inf)
                previous[0] = 0.0

                for row in range(1, m + 1):
                    current = np.full(m + 1, np.inf)
                    start = max(1, row - radius)
                    end = min(m, row + radius)

                    for col in range(start, end + 1):
                        cost = (query[row - 1] - candidate[col - 1]) ** 2
                        current[col] = cost + min(
                            previous[col],
                            current[col - 1],
                            previous[col - 1],
                        )

                    previous = current

                distance = previous[m]
                dist_profile[i] = distance

                if distance < bsf:
                    bsf = distance

            if not np.isfinite(bsf):
                break

            best_index = int(np.argmin(dist_profile))
            bestmatch["index"].append(best_index)
            bestmatch["distance"].append(float(dist_profile[best_index]))

            left = max(0, best_index - excl_zone)
            right = min(N, best_index + excl_zone + 1)
            active[left:right] = False

        return bestmatch
