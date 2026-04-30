#!/usr/bin/env python
"""Entry point for the masked-imputation preset."""

import sys

sys.path.insert(0, ".")

from experiments.ablations.posterior_strategy_ablation import main


if __name__ == "__main__":
    main(default_preset="imputation")
