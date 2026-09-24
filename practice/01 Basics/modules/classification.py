import numpy as np

from modules.metrics import *
from modules.utils import z_normalize
from typing import Self


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier

    Parameters
    ----------
    n_neighbors: number of neighbors
    metric: distance measure between time series
             Options: {euclidean, dtw}
    metric_params: dictionary containing parameters for the distance metric being used
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:

        self.n_neighbors: int = n_neighbors
        self.metric: str = metric
        self.metric_params: dict | None = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        """
        Fit the model using X_train as training data and Y_train as labels

        Parameters
        ----------
        X_train: train set with shape (ts_number, ts_length)
        Y_train: labels of the train set
        
        Returns
        -------
        self: the fitted model
        """
       
        self.X_train = X_train
        self.Y_train = Y_train

        return self


    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """
        Compute distance between the train and test samples
        """
        # Применяем z-нормализацию, если это указано в параметрах метрики
        if self.metric_params.get('normalize', False):
            x_train = z_normalize(x_train)
            x_test = z_normalize(x_test)

        if self.metric == 'euclidean':
            # Если передана нормализация, можно использовать norm_ED_distance или ED_distance на нормированных
            if self.metric_params.get('normalize', False):
                dist = ED_distance(x_train, x_test)
            else:
                dist = ED_distance(x_train, x_test)
        elif self.metric == 'dtw':
            # Извлекаем параметр r для DTW, если он задан
            r = self.metric_params.get('r', None)
            if r is not None:
                dist = DTW_distance(x_train, x_test, r=r)
            else:
                dist = DTW_distance(x_train, x_test)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return float(dist)

    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """
        Find the k nearest neighbors of the test sample
        """
        distances = []
        for i in range(len(self.X_train)):
            d = self._distance(self.X_train[i], x_test)
            distances.append((d, self.Y_train[i]))

        # Сортируем по возрастанию расстояния
        distances.sort(key=lambda x: x[0])

        # Берем k ближайших соседей
        neighbors = distances[:self.n_neighbors]

        return neighbors

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for samples of the test set
        """
        y_pred = []

        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)
            # Извлекаем метки классов соседей
            labels = [label for _, label in neighbors]

            # Находим наиболее часто встречающийся класс (мажоритарное голосование)
            predicted_class = max(set(labels), key=labels.count)
            y_pred.append(predicted_class)

        return np.array(y_pred)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score

    Parameters
    ----------
    y_true: ground truth (correct) labels
    y_pred: predicted labels returned by a classifier

    Returns
    -------
    score: accuracy classification score
    """

    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score
