import argparse
import json
from pathlib import Path

import numpy as np

from src.data.energy_scheduling import (
    energy_split_indices,
    load_energy_instance,
    load_energy_price_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/Energy/SchedulingInstances"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/rebuttal/energy_scheduling_task_start/pretraining_structure"
)
OUTPUT_NAME = "realized_taskwise_invariant_variation.json"


def task_start_components(instance):
    return tuple(
        (task, start)
        for task in range(instance.n_tasks)
        for start in range(
            int(instance.earliest[task]),
            int(instance.latest[task] - instance.durations[task] + 1),
        )
    )


def component_load_matrix(instance, components):
    matrix = np.zeros((len(components), instance.horizon), dtype=float)
    hours_per_slot = float(instance.q_minutes) / 60
    for index, (task, start) in enumerate(components):
        stop = start + int(instance.durations[task])
        matrix[index, start:stop] = instance.powers[task] * hours_per_slot
    return matrix


def taskwise_constant_basis(components, n_tasks):
    component_tasks = np.array([task for task, _ in components])
    counts = np.bincount(component_tasks, minlength=n_tasks)
    if np.any(counts == 0):
        raise ValueError("each task must have a feasible start")
    return (component_tasks[:, None] == np.arange(n_tasks)) / np.sqrt(counts)


def realized_invariant_projection(realized_component_costs, invariant_basis):
    costs = np.asarray(realized_component_costs, dtype=float)
    basis = np.asarray(invariant_basis, dtype=float)
    if costs.ndim != 2:
        raise ValueError("realized_component_costs must be two-dimensional")
    if basis.ndim != 2 or basis.shape[0] != costs.shape[1]:
        raise ValueError(f"invariant_basis must have shape ({costs.shape[1]}, k)")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
        raise ValueError("invariant_basis columns must be orthonormal")

    centered = costs - costs.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum())
    invariant = float(np.square(centered @ basis).sum())
    fraction = (
        float("nan")
        if total <= np.finfo(float).eps
        else float(np.clip(invariant / total, 0, 1))
    )
    return {
        "centered_realized_cost_variation": total,
        "guaranteed_invariant_realized_cost_variation": invariant,
        "guaranteed_invariant_realized_variation_fraction": fraction,
    }


def analyze_structure(
    data_root=DEFAULT_DATA_ROOT,
    energy_instance="load3",
    validation_fraction=0.5,
):
    data_root = Path(data_root).expanduser().resolve()
    dataset = load_energy_price_dataset(
        data_root,
        validation_fraction=validation_fraction,
    )
    instance = load_energy_instance(data_root / energy_instance / "day01.txt")
    start, stop = energy_split_indices(
        len(dataset.slot_costs),
        split="validation",
        validation_fraction=validation_fraction,
    )
    slot_costs = np.asarray(dataset.slot_costs[start:stop], dtype=float)
    components = task_start_components(instance)
    component_costs = (
        slot_costs
        @ component_load_matrix(
            instance,
            components,
        ).T
    )
    basis = taskwise_constant_basis(components, instance.n_tasks)
    return {
        "validation_start": int(start),
        "validation_stop": int(stop),
        "n_days": int(stop - start),
        "n_available_days": int(len(dataset.slot_costs)),
        "n_tasks": int(instance.n_tasks),
        "n_task_start_components": len(components),
        "invariant_subspace_dimension": int(basis.shape[1]),
        **realized_invariant_projection(component_costs, basis),
    }


def write_result(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--energy-instance", default="load3")
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_structure(
        data_root=args.data_root,
        energy_instance=args.energy_instance,
        validation_fraction=args.validation_fraction,
    )
    write_result(result, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
