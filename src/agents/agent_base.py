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

    def _init_phase_buffer(self, max_buffer_size: Optional[int] = None) -> None:
        self.max_buffer_size = max_buffer_size
        self._phase_buffer_X = []
        self._phase_buffer_Y = []
        self._phase_buffer_sel = []
        self._phase_buffer_prob = []
        self._phase_buffer_obs = []

    def _append_phase_buffer(
        self,
        X,
        rewards,
        selection,
        prob_values=None,
        obs_mask=None,
    ) -> None:
        if self.max_buffer_size is not None and self.max_buffer_size > 0:
            if len(self._phase_buffer_X) >= self.max_buffer_size:
                self._phase_buffer_X.pop(0)
                self._phase_buffer_Y.pop(0)
                self._phase_buffer_sel.pop(0)
                self._phase_buffer_prob.pop(0)
                self._phase_buffer_obs.pop(0)

        self._phase_buffer_X.append(X)
        self._phase_buffer_Y.append(rewards)
        self._phase_buffer_sel.append(selection)
        self._phase_buffer_prob.append(prob_values)
        self._phase_buffer_obs.append(obs_mask)

    def _clear_phase_buffer(self) -> None:
        self._phase_buffer_X.clear()
        self._phase_buffer_Y.clear()
        self._phase_buffer_sel.clear()
        self._phase_buffer_prob.clear()
        self._phase_buffer_obs.clear()
