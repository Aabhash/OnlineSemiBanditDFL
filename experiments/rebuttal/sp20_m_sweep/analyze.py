import csv, json, pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / 'results/rebuttal/sp20_m_sweep'
ARCHIVE = REPO / 'results/paper/eval_online_10k/metrics'
NONLINEAR = REPO / 'results/rebuttal/camera_ready_nonlinear/ordered_nonlinear/results.json'
DATASET = 'shortest_path_20x20'
METHOD = 'BayesianSPO'
KEY = f'{METHOD}_{DATASET}_semi_bandit'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def mean_sd(values):
    return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}


def main():
    config = json.loads((RUN / 'config.json').read_text())
    seeds = config['seeds']
    runs = {(m, seed): json.loads((RUN / f'M{m}/seed_{seed}/result.json').read_text())
            for m in config['m_values'] for seed in seeds}
    spo = np.array([runs[2, seed]['final_regret'] for seed in seeds])
    summaries = json.loads((ARCHIVE / 'results_summary.json').read_text())
    regrets = json.loads((ARCHIVE / 'final_eval_regrets.json').read_text())
    with (ARCHIVE / 'final_eval_summary.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        columns, table = reader.fieldnames, list(reader)
    with (ARCHIVE / 'trajectories.pkl').open('rb') as f:
        archived_trajectories = pickle.load(f)

    summaries[KEY].update(cumulative_regrets=spo.tolist(),
                          mean_cumulative_regret=float(spo.mean()),
                          std_cumulative_regret=float(spo.std(ddof=1)),
                          min_cumulative_regret=float(spo.min()),
                          max_cumulative_regret=float(spo.max()))
    regrets[DATASET][METHOD] = spo.tolist()
    for row in table:
        if (row['Dataset'], row['Agent'], row['Feedback Type']) == (DATASET, METHOD, 'semi_bandit'):
            row.update({'Mean Regret': float(spo.mean()), 'Std Regret': float(spo.std(ddof=1))})

    overrides = {}
    for seed in seeds:
        key = METHOD, DATASET, 'semi_bandit', seed
        with np.load(RUN / f'M2/seed_{seed}/trajectory.npz') as data:
            trajectory = {name: data[name].tolist() for name in data.files}
        trajectory['wall_time_seconds'] = archived_trajectories[key]['wall_time_seconds']
        trajectory['hyperparameters'] = runs[2, seed]['params']
        overrides[key] = trajectory

    mse = np.array(summaries[f'BayesianMSE_{DATASET}_semi_bandit']['cumulative_regrets'])
    pfy = summaries[f'BayesianPFY_{DATASET}_semi_bandit']['mean_cumulative_regret']
    test = wilcoxon(spo, mse, alternative='less', method='exact')
    comparison = {
        'baseline': 'Table 2 with the documented-config SP20 BayesianSPO rerun',
        'source': str(RUN.relative_to(REPO)),
        'seeds': seeds,
        'rounds': config['rounds'],
        'wall_time_policy': 'Retain historical evaluation timings; fresh rerun timings remain in each result.json.',
        'spo': mean_sd(spo),
        'mse': mean_sd(mse),
        'spo_reduction_vs_mse_percent': float(100 * (mse.mean() - spo.mean()) / mse.mean()),
        'pfy_gap_vs_spo_percent': float(100 * (pfy - spo.mean()) / spo.mean()),
        'paired_wins_vs_mse': int(np.sum(spo < mse)),
        'wilcoxon_spo_less_mse': {'statistic': float(test.statistic), 'pvalue': float(test.pvalue)},
        'm_sweep': {str(m): mean_sd([runs[m, seed]['final_regret'] for seed in seeds])
                    for m in sorted(config['m_values'])},
    }
    nonlinear = json.loads(NONLINEAR.read_text())
    comparison['nonlinear_reduction_vs_table2_percent'] = {
        table['dataset']: {
            method: 100 * (1 - table['estimates'][f'{method}_nonlinear']['mean']
                           / summaries[f'{method}_{table["dataset"]}_semi_bandit']['mean_cumulative_regret'])
            for method in ('BayesianSPO', 'BayesianMSE', 'BLL-Online')}
        for table in nonlinear['tables']}

    diagnostic = {'round_start_exclusive': config['rounds'] - 2000, 'round_end': config['rounds']}
    for metric in ('contested_mae', 'noncontested_mae'):
        values = {}
        for method in ('BayesianSPO', 'BayesianMSE'):
            source = overrides if method == 'BayesianSPO' else archived_trajectories
            per_seed = []
            for seed in seeds:
                data = source[method, DATASET, 'semi_bandit', seed]
                mask = np.asarray(data['diagnostic_rounds']) > diagnostic['round_start_exclusive']
                per_seed.append(np.nanmean(np.asarray(data[metric])[mask]))
            values[method] = float(np.mean(per_seed))
        values['spo_reduction_percent'] = 100 * (1 - values['BayesianSPO'] / values['BayesianMSE'])
        diagnostic[metric] = values
    comparison['sp20_mae'] = diagnostic

    cumulative = np.mean([overrides[METHOD, DATASET, 'semi_bandit', seed]['cumulative_regret']
                          for seed in seeds], axis=0)
    indices = sorted(set(range(0, len(cumulative), max(1, len(cumulative) // 500))) | {len(cumulative) - 1})
    rounds, values = np.array(indices) + 1, cumulative[indices]
    mask = (rounds >= 1000) & (values > 0)
    comparison['sp20_loglog_slope_1k_to_10k'] = float(np.polyfit(np.log(rounds[mask]), np.log(values[mask]), 1)[0])

    output = RUN / 'metrics'
    output.mkdir(exist_ok=True)
    write_json(output / 'results_summary.json', summaries)
    write_json(output / 'final_eval_regrets.json', regrets)
    write_json(output / 'comparison.json', comparison)
    with (output / 'final_eval_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    with (output / 'trajectory_overrides_sp20.pkl').open('wb') as f:
        pickle.dump(overrides, f)
    with (RUN / 'imputation_samples.pkl').open('wb') as f:
        pickle.dump({(f'M{m}', seed): {'final_regret': run['final_regret']}
                     for (m, seed), run in runs.items()}, f)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
