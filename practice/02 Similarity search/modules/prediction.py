import numpy as np
import math

from modules.bestmatch import UCR_DTW, topK_match
import mass_ts as mts


default_match_alg_params = {
    'UCR-DTW': {
        'topK': 3,
        'r': 0.05,
        'excl_zone_frac': 1,
        'is_normalize': True
    },
    'MASS': {
        'topK': 3,
        'excl_zone_frac': 1
    }
}


class BestMatchPredictor:
    """
    Predictor based on best match algorithm
    """

    def __init__(self, h: int = 1, match_alg: str = 'UCR-DTW', match_alg_params: dict | None = None, aggr_func: str = 'average') -> None:
        """ 
        Constructor of class BestMatchPredictor

        Parameters
        ----------    
        h: prediction horizon
        match_algorithm: name of the best match algorithm
        match_algorithm_params: input parameters for the best match algorithm
        aggr_func: aggregate function
        """

        self.h: int = h
        self.match_alg: str = match_alg
        self.match_alg_params: dict | None = default_match_alg_params[match_alg].copy()
        if match_alg_params is not None:
            self.match_alg_params.update(match_alg_params)
        self.agg_func: str = aggr_func


    def _calculate_predict_values(self, topK_subs_predict_values: np.array) -> np.ndarray:
        """
        Calculate the future values of the time series using the aggregate function

        Parameters
        ----------
        topK_subs_predict_values: values of time series, which are located after topK subsequences

        Returns
        -------
        predict_values: prediction values
        """

        match self.agg_func:
            case 'average':
                predict_values = topK_subs_predict_values.mean(axis=0).round()
            case 'median':
                predict_values = topK_subs_predict_values.median(axis=0).round()
            case _:
                raise NotImplementedError
        
        return predict_values

    def predict(self, ts: np.ndarray, query: np.ndarray) -> np.array:
        """
        Predict time series at future horizon

        Parameters
        ----------
        ts: time series
        query: query, shorter than time series

        Returns
        -------
        predict_values: prediction values
        """

        predict_values = np.zeros((self.h,))
        m = len(query)
        topK = self.match_alg_params['topK']

        # 1. Запуск выбранного алгоритма поиска похожих подпоследовательностей
        if self.match_alg == 'UCR-DTW':
            finder = UCR_DTW(
                excl_zone_frac=self.match_alg_params.get('excl_zone_frac', 1.0),
                topK=topK,
                is_normalize=self.match_alg_params.get('is_normalize', True),
                r=self.match_alg_params.get('r', 0.1)
            )
            results = finder.perform(ts, query)
            indices = results.get('index', results.get('indices', []))

        elif self.match_alg == 'MASS':
            # Вычисляем профиль расстояний с помощью библиотеки mass_ts
            dist_profile = mts.mass2(ts, query)

            # Конвертируем относительную зону исключения (frac) в количество элементов (int)
            excl_zone_frac = self.match_alg_params.get('excl_zone_frac', 1.0)
            excl_zone = int(excl_zone_frac * m)

            # Находим topK совпадений с помощью вашей функции topK_match
            results = topK_match(
                dist_profile=dist_profile,
                excl_zone=excl_zone,
                topK=topK
            )
            indices = results.get('indices', [])
        else:
            raise NotImplementedError(f"Алгоритм {self.match_alg} не поддерживается.")

        # 2. Сбор будущих отрезков (длиной h), идущих после найденных совпадений
        future_segments = []
        for idx in indices[:topK]:
            start_idx = idx + m
            end_idx = start_idx + self.h

            if end_idx <= len(ts):
                future_segments.append(ts[start_idx:end_idx])

        if not future_segments:
            return np.zeros((self.h,))

        # 3. Превращаем список в матрицу и агрегируем прогноз
        topK_subs_predict_values = np.array(future_segments)
        predict_values = self._calculate_predict_values(topK_subs_predict_values)

        return predict_values