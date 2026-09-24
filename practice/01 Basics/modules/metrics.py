import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = np.linalg.norm(ts1 - ts2)

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)

    # Вычисление среднего арифметического mu_T
    mu_1 = np.mean(ts1)
    mu_2 = np.mean(ts2)

    # Вычисление стандартного отклонения sigma_T по формуле из задания: sqrt(mean(t^2) - mu^2)
    sigma_1 = np.sqrt(np.mean(ts1 ** 2) - mu_1 ** 2)
    sigma_2 = np.sqrt(np.mean(ts2 ** 2) - mu_2 ** 2)

    # Защита от деления на ноль, если ряд константный
    if sigma_1 == 0:
        sigma_1 = 1e-8
    if sigma_2 == 0:
        sigma_2 = 1e-8

    # Скалярное произведение <T1, T2>
    dot_product = np.dot(ts1, ts2)

    # Вычисление подкоренного выражения
    numerator = dot_product - n * mu_1 * mu_2
    denominator = n * sigma_1 * sigma_2

    inner_val = 2 * n * (1.0 - (numerator / denominator))

    # Используем abs под корнем на случай мелких погрешностей округления с плавающей точкой
    norm_ed_dist = np.sqrt(np.abs(inner_val))

    return float(norm_ed_dist)


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n, m = len(ts1), len(ts2)

    # Расчет ширины окна (Sakoe-Chiba window)
    if r <= 1.0:
        window = int(max(r * max(n, m), abs(n - m)))
    else:
        window = int(r)

    # Инициализируем матрицу расстояний (бесконечности)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        # Ограничения для окна поиска
        window_start = max(1, i - window)
        window_end = min(m + 1, i + window + 1)

        for j in range(window_start, window_end):
            # Стоимость перехода (абсолютная разница или квадрат разницы)
            cost = (ts1[i-1] - ts2[j-1]) ** 2

            # Находим минимум среди трех путей (вставка, удаление, совпадение)
            min_prev = min(
                dtw_matrix[i - 1, j],  # вставка
                dtw_matrix[i, j - 1],  # удаление
                dtw_matrix[i - 1, j - 1]  # совпадение
            )
            dtw_matrix[i, j] = cost + min_prev

    # Возвращаем итоговое расстояние из правого нижнего угла
    dtw_dist = float(dtw_matrix[n, m])

    return dtw_dist
