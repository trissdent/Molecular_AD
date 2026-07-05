from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import zoom

class BaseTransformer(ABC):
    """
    Base transformer class.
    Modify for your task.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MRITransformer(BaseTransformer):

    def __init__(self, target_shape: Tuple[int, int, int] = (128, 128, 128), margin: int = 5):
        self.target_shape = target_shape
        self.margin = margin

    def _crop_to_brain(self, data: np.ndarray) -> np.ndarray:
        nonzero = np.nonzero(data)
        if len(nonzero[0]) == 0:
            return data

        mins = np.array([n.min() for n in nonzero])
        maxs = np.array([n.max() for n in nonzero])

        lengths = maxs - mins
        max_len = lengths.max()
        cube_size = max_len + 2 * self.margin
        centers = (mins + maxs) // 2

        starts = np.clip(centers - cube_size // 2, 0, np.array(data.shape) - 1).astype(int)
        ends = (starts + cube_size).astype(int)
        for i in range(3):
            if ends[i] > data.shape[i]:
                ends[i] = data.shape[i]
                starts[i] = max(0, ends[i] - cube_size)

        return data[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    def _resize(self, data: np.ndarray) -> np.ndarray:
        zoom_factors = [t / s for t, s in zip(self.target_shape, data.shape)]
        return zoom(data, zoom_factors, order=1)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            return (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        volume = self._crop_to_brain(volume)
        volume = self._resize(volume)
        volume = self._normalize(volume)
        return volume

