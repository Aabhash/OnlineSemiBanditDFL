import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod


class BaseValidator(ABC):

    @abstractmethod
    def evaluate(self, agent) -> float:
        pass

    @abstractmethod
    def get_details(self, agent) -> Dict[str, float]:
        pass


class OnlineValidator(BaseValidator):

    def __init__(self, env, n_samples: int, seed: int):
        self.n_samples = n_samples
        self.solver = env.solve
        self.weights = env.weights
        self.capacity = env.capacity
        self.per_arm_context = getattr(env, 'per_arm_context', False)

        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        self.X_val, self.Y_val = env.generate_test_set(n_samples)

    def evaluate(self, agent) -> float:
        total_regret = 0.0

        for i in range(self.n_samples):
            X = self.X_val[i]
            Y = self.Y_val[i]

            selection = agent.act(X)
            selection = np.asarray(selection)

            optimal = self.solver(Y)
            optimal = np.asarray(optimal)

            agent_reward = np.sum(Y * selection)
            optimal_reward = np.sum(Y * optimal)
            regret = optimal_reward - agent_reward

            total_regret += regret

        return total_regret

    def get_details(self, agent) -> Dict[str, float]:
        total_reward = 0.0
        total_optimal = 0.0

        for i in range(self.n_samples):
            X = self.X_val[i]
            Y = self.Y_val[i]

            selection = agent.act(X)
            selection = np.asarray(selection)

            optimal = self.solver(Y)
            optimal = np.asarray(optimal)

            total_reward += np.sum(Y * selection)
            total_optimal += np.sum(Y * optimal)

        total_regret = total_optimal - total_reward

        return {
            'regret': total_regret,
            'total_reward': total_reward,
            'total_optimal': total_optimal,
            'avg_reward': total_reward / self.n_samples,
            'avg_optimal': total_optimal / self.n_samples,
            'avg_regret': total_regret / self.n_samples,
            'pct_optimal': 100 * total_reward / total_optimal if total_optimal > 0 else 0,
            'n_samples': self.n_samples,
        }


class OfflineValidator(BaseValidator):

    def __init__(self, env):
        self.env = env

        if not hasattr(env, 'evaluate_on_validation'):
            raise ValueError(
                f"Environment {type(env).__name__} does not support validation evaluation"
            )

        if hasattr(env, 'validation_weeks') and env.validation_weeks == 0:
            raise ValueError("No validation weeks configured")

    def evaluate(self, agent) -> float:
        result = self.env.evaluate_on_validation(agent)
        if not result:
            raise ValueError("No validation data available from environment")
        return result['regret']

    def get_details(self, agent) -> Dict[str, float]:
        return self.env.evaluate_on_validation(agent)


class HoldoutValidator(BaseValidator):

    def __init__(self, env):
        self.env = env

        if not hasattr(env, 'evaluate_on_holdout'):
            raise ValueError(
                f"Environment {type(env).__name__} does not support holdout evaluation"
            )

    def evaluate(self, agent) -> float:
        result = self.env.evaluate_on_holdout(agent)
        if not result:
            raise ValueError("No holdout data available from environment")
        return result['regret']

    def get_details(self, agent) -> Dict[str, float]:
        return self.env.evaluate_on_holdout(agent)


def create_validator(env, config: Dict[str, Any]) -> BaseValidator:
    validation_config = config['validation']
    n_samples = validation_config['n_val_samples']
    seed = validation_config['seed']
    return OnlineValidator(env, n_samples=n_samples, seed=seed)


def get_validation_type(env) -> str:
    has_validation_weeks = (
        hasattr(env, 'validation_weeks') and
        env.validation_weeks > 0
    )

    if has_validation_weeks:
        return 'offline'
    return 'online'


class ValidationResult:

    def __init__(
        self,
        regret: float,
        details: Dict[str, float],
        round_num: int,
        validation_type: str,
    ):
        self.regret = regret
        self.details = details
        self.round_num = round_num
        self.validation_type = validation_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regret': self.regret,
            'round': self.round_num,
            'validation_type': self.validation_type,
            **self.details,
        }
