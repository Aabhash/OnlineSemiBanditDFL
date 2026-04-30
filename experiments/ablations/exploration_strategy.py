#!/usr/bin/env python
"""Compatibility wrapper for the exploration preset."""

import sys

sys.path.insert(0, ".")

from experiments.ablations.posterior_strategy_ablation import (
    STRATEGY_LABELS,
    load_tuned_params,
    main,
    resolve_base_hp,
)


if __name__ == "__main__":
    main(default_preset="exploration")
