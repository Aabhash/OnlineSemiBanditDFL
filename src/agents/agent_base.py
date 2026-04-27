from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class AgentBase(ABC):

    @abstractmethod
    def act(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, X: np.ndarray, selection: np.ndarray,
               rewards: np.ndarray, lr: float = None,
               full_feedback: bool = False, **kwargs) -> None:
        ...

    def predict_mean(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    def predict_uncertainty(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None
