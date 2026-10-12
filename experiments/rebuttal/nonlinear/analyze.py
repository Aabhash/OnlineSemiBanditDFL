import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.rebuttal.nonlinear.adapters import write_json

ACCEPTED = REPO / 'results/rebuttal/camera_ready_nonlinear/ordered_nonlinear/results.json'
COMPARISONS = {'nonlinear_spo_vs_linear_spo': (('BayesianSPO', 'linear'), ('BayesianSPO', 'nonlinear')),
               'nonlinear_mse_vs_linear_mse': (('BayesianMSE', 'linear'), ('BayesianMSE', 'nonlinear')),
               'linear_spo_vs_mse': (('BayesianMSE', 'linear'), ('BayesianSPO', 'linear')),
               'nonlinear_spo_vs_mse': (('BayesianMSE', 'nonlinear'), ('BayesianSPO', 'nonlinear')),
               'nonlinear_bll_vs_linear_bll': (('BLL-Online', 'linear'), ('BLL-Online', 'nonlinear'))}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
            'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None}


def paired_summary(pairs):
    before = np.asarray([row['before_gap'] for row in pairs], dtype=float)
    after = np.asarray([row['after_gap'] for row in pairs], dtype=float)
    reductions = before - after
    result = {'n': len(pairs), 'seeds': [row['seed'] for row in pairs],
              'wins': int(np.sum(reductions > 0)), 'ties': int(np.sum(reductions == 0)), 'losses': int(np.sum(reductions < 0)),
              'before': mean_sd(before), 'after': mean_sd(after), 'gap_reduction': mean_sd(reductions), 'paired_rows': pairs,
              'gap_reduction_ci95': None, 'ratio_of_means_reduction_percent': None, 'ratio_of_means_reduction_percent_ci95': None}
    if len(pairs) and before.mean() > 0:
        result['ratio_of_means_reduction_percent'] = float(100 * reductions.mean() / before.mean())
    if len(pairs) < 2: return result
    indices = np.random.default_rng(20261009).integers(0, len(pairs), size=(20000, len(pairs)))
    boot_reductions, boot_before = reductions[indices].mean(axis=1), before[indices].mean(axis=1)
    result['gap_reduction_ci95'] = np.quantile(boot_reductions, [.025, .975]).tolist()
    if np.all(boot_before > 0):
        result['ratio_of_means_reduction_percent_ci95'] = np.quantile(100 * boot_reductions / boot_before, [.025, .975]).tolist()
    return result


def summarize(output, config):
    output = Path(output).resolve()
    if output == ACCEPTED.parent or output.is_relative_to(ACCEPTED.parent / 'sources'):
        raise ValueError('Accepted results and source archives are read-only')
    rows = []
    for seed in config['seeds']:
        for dataset in config['dataset_order']:
            for variant in config['passes']:
                path = output / 'seeds' / f'seed_{seed}' / 'datasets' / dataset / 'passes' / variant / 'complete.json'
                if path.exists(): rows.extend(json.loads(path.read_text())['rows'])
    tables = []
    for dataset in config['dataset_order']:
        raw = [row for row in rows if row['dataset'] == dataset]
        keyed = {(row['seed'], row['method'], row['variant']): row for row in raw}
        estimates = {f'{method}_{variant}': mean_sd([row['final_regret'] for row in raw if row['method'] == method and row['variant'] == variant])
                     for method in config['datasets'][dataset]['method_order'] for variant in config['passes']}
        comparisons = {}
        for name, (before, after) in COMPARISONS.items():
            if any(method not in config['target_methods'] for method, _ in (before, after)): continue
            pairs = []
            for seed in config['seeds']:
                old, new = keyed.get((seed, *before)), keyed.get((seed, *after))
                if old is not None and new is not None:
                    pairs.append({'seed': seed, 'before_gap': old['final_regret'], 'after_gap': new['final_regret'],
                                  'before_flagged': old['flagged'], 'after_flagged': new['flagged']})
            comparisons[name] = paired_summary(pairs)
        tables.append({'dataset': dataset, 'estimates': estimates, 'comparisons': comparisons, 'raw_endpoints': raw})
    expected_methods, expected_cells = config['expected_method_runs'], config['expected_target_cells']
    result = {'experiment': 'ordered_nonlinear', 'seeds': config['seeds'], 'rounds': config['rounds'], 'passes': config['passes'],
              'methods': config['target_methods'], 'completed_cells': sum(row['target'] for row in rows), 'expected_cells': expected_cells,
              'completed_method_runs': len(rows), 'expected_method_runs': expected_methods,
              'complete': len(rows) == expected_methods, 'tables': tables,
              'bootstrap': {'draws': 20000, 'seed': 20261009, 'interval': 'Pointwise paired-seed percentile 95%; no multiplicity adjustment'},
              'retention_policy': 'All completed endpoints and solver flags are retained, including infeasible runs.',
              'flagged_method_runs': sum(row['flagged'] for row in rows)}
    write_json(output / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ACCEPTED, help='Saved JSON to display, or a run directory to summarize')
    args = parser.parse_args()
    if args.input.is_file():
        result = json.loads(args.input.read_text())
    else:
        result = summarize(args.input, json.loads((args.input / 'config.json').read_text()))
    print(f'{result["completed_cells"]} target cells; {args.input}')


if __name__ == '__main__':
    main()
