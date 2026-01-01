from src.envs import (
    ConstrainedBanditEnv,
    SyntheticKnapsackEnv,
    ShortestPathEnv,
    CriteoEnv,
    FantasyEnv,
    create_env,
)

from src.data import (
    SyntheticKnapsackData,
    SyntheticGridData,
    FantasyData,
)

from src.models import (
    LinearModel,
    MLPModel,
    BayesianLinearModel,
    create_model,
)

from src.agents import (
    # Shared context - main contribution
    BayesianSPOAgent,
    # Shared context - baselines
    RandomAgent,
    GreedyAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    LinUCBAgent,
    NeuralUCBAgent,
    EpsilonGreedyAgent,
    MSENeuralAgent,
    run_experiment,
    # Per-arm context - our main contribution
    PerArmBayesianSPOAgent,
    # Per-arm context - baselines
    CNUCBAgent,
    CNTSAgent,
    PerArmLinUCBAgent,
    PerArmThompsonSamplingAgent,
    PerArmEpsilonGreedyAgent,
    PerArmMSENeuralAgent,
    PerArmRandomAgent,
)

from src.solver import solve_knapsack

from src.losses import (
    spo_plus_grad,
    spo_plus_grad_balanced,
    spo_plus_grad_bandit,
    nce_grad_bandit,
)

__all__ = [
    # Environments
    "ConstrainedBanditEnv",
    "SyntheticKnapsackEnv",
    "ShortestPathEnv",
    "CriteoEnv",
    "FantasyEnv",
    "create_env",
    # Data
    "SyntheticKnapsackData",
    "SyntheticGridData",
    "FantasyData",
    # Models
    "LinearModel",
    "MLPModel",
    "BayesianLinearModel",
    "create_model",
    # Shared context agents
    "BayesianSPOAgent",
    "RandomAgent",
    "GreedyAgent",
    "ThompsonSamplingAgent",
    "UCBAgent",
    "LinUCBAgent",
    "NeuralUCBAgent",
    "EpsilonGreedyAgent",
    "MSENeuralAgent",
    "run_experiment",
    # Per-arm context agents
    "PerArmBayesianSPOAgent",
    "CNUCBAgent",
    "CNTSAgent",
    "PerArmLinUCBAgent",
    "PerArmThompsonSamplingAgent",
    "PerArmEpsilonGreedyAgent",
    "PerArmMSENeuralAgent",
    "PerArmRandomAgent",
    # Solver
    "solve_knapsack",
    # Losses
    "spo_plus_grad",
    "spo_plus_grad_balanced",
    "spo_plus_grad_bandit",
    "nce_grad_bandit",
]
