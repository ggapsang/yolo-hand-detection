from abc import ABC, abstractmethod

import numpy as np


class AbstractDetector(ABC):

    @abstractmethod
    def inference(self, image: np.ndarray) -> tuple:
        """
        Returns:
            (width, height, inference_time, results)
            results: list of (class_id, label, confidence, x, y, w, h)
        """
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """현재 사용 중인 backend 이름"""
        ...
