import argparse, json, pickle, subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import numpy as np
import pandas as pd
from lets_plot import (
    aes, coord_cartesian, element_blank, element_line, element_text, geom_bar, geom_errorbar, geom_line,
    geom_point, geom_ribbon, geom_text, gggrid, ggplot, ggsize, guide_legend, guides, labs,
    scale_color_manual, scale_fill_manual, scale_linetype_manual, scale_shape_manual,
    scale_x_continuous, scale_y_continuous, scale_y_discrete, theme, theme_minimal,
)
from lets_plot.export import ggsave

from experiments.plot_paper_core_results import PAPER_EMPHASIS_FONT, PAPER_FONT, SHAPES


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'results/rebuttal/energy_scheduling_task_start/semi_bandit_all_methods_30seed'
METHODS = ['BayesianSPO', 'BayesianMSE', 'LinUCB', 'BLL-Online', 'TS', 'EpsilonGreedy']
LABELS = ['BayesianSPO', 'BayesianMSE', 'LinUCB', 'BLL-Online', 'TS', 'ε-Greedy']


def style_bar_svg(svg):
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns)
    root = ET.fromstring(svg.read_text())
    bars = [(parent, child) for parent in root.iter() for child in parent
            if child.tag == f'{{{ns}}}rect' and child.get('fill') == 'rgb(242,242,242)']
    # Overlay one shared vector hatch, leaving the solid background and borders intact.
    defs = ET.SubElement(root, f'{{{ns}}}defs')
    pattern = ET.SubElement(defs, f'{{{ns}}}pattern', id='bar-hatch', width='8', height='8',
                            patternUnits='userSpaceOnUse')
    ET.SubElement(pattern, f'{{{ns}}}path', d='M-2 2L2 -2M0 8L8 0M6 10L10 6',
                  fill='none', stroke='#999999', attrib={'stroke-width': '0.7'})
    for parent, bar in bars:
        ET.SubElement(parent, f'{{{ns}}}rect', attrib=bar.attrib | {'fill': 'url(#bar-hatch)'})
    svg.write_text(ET.tostring(root, encoding='unicode'))


def plot_results(source, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    curves, endpoints, markers = [], [], []
    for index, method in enumerate(METHODS):
        directory = REPO / manifest['source_dirs'][method]
        with (directory / 'trajectories.pkl').open('rb') as handle:
            trajectories = pickle.load(handle)
        scale = json.loads((directory / 'manifest.json').read_text())['benchmark']['cost_scale']
        values = np.array([trajectories[(method, seed)]['cumulative_regret']
                           for seed in manifest['seeds']]) * scale / 1e6
        values = np.column_stack([np.zeros(len(values)), values])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        endpoints.append({'method': method, 'mean': mean[-1], 'low': mean[-1] - sd[-1],
                          'high': mean[-1] + sd[-1], 'label_x': mean[-1] + sd[-1] + 3,
                          'label': f'{mean[-1]:.1f}'})
        curves.extend({'method': method, 'day': day, 'mean': avg, 'low': avg - spread, 'high': avg + spread}
                      for day, (avg, spread) in enumerate(zip(mean, sd)))
        markers.extend({'method': method, 'day': day, 'mean': mean[day]}
                       for day in range(50 + 8 * index, len(mean), 65))

    style = theme_minimal() + theme(
        text=element_text(family=PAPER_FONT, size=20, color='#000000'),
        axis_title=element_text(size=22),
        axis_ticks=element_line(color='#7A7A7A', size=1.0), axis_ticks_length=5,
        axis_text_spacing=4,
        panel_grid_major=element_line(color='#E4E4E4', size=0.62),
        panel_grid_minor=element_blank(), legend_title=element_blank(),
        legend_position='top', plot_tag=element_text(family=PAPER_EMPHASIS_FONT, size=22),
        plot_tag_location='margin',
    )
    palette = dict(zip(METHODS, ['#92AFBF', '#B99D88', '#A4A4A4', '#9BB8AD', '#C1B28E', '#AF9DBC']))
    colors = scale_color_manual(values=[palette[method] for method in METHODS], breaks=METHODS, labels=LABELS)
    trajectory = (
        ggplot(pd.DataFrame(curves), aes(x='day', y='mean', color='method'))
        + geom_ribbon(aes(ymin='low', ymax='high', fill='method'), alpha=0.10,
                      color='transparent', show_legend=False)
        + geom_line(aes(linetype='method'), size=1.25)
        + geom_point(aes(shape='method'), data=pd.DataFrame(markers), size=4, stroke=1.0)
        + colors + scale_fill_manual(values=[palette[method] for method in METHODS], breaks=METHODS)
        + scale_shape_manual(values=SHAPES, breaks=METHODS, labels=LABELS)
        + scale_linetype_manual(values=['solid', 'dashed', 'dotdash', 'dotted', 'longdash', 'twodash'],
                                breaks=METHODS, labels=LABELS)
        + scale_x_continuous(breaks=[0, 100, 200, 300, 395], limits=[0, 395])
        + scale_y_continuous(breaks=[0, 50, 100, 150, 175], expand=[0, 0])
        + coord_cartesian(ylim=[0, 175])
        + labs(x='Rounds', y='Cumulative regret (M)', tag='(a)')
        + guides(color=guide_legend(ncol=6), shape=guide_legend(ncol=6), linetype=guide_legend(ncol=6)) + style
    )
    endpoints = pd.DataFrame(endpoints)
    final_regret = (
        ggplot(endpoints, aes(x='mean', y='method'))
        + geom_bar(color='#686868', fill='#F2F2F2', size=1.05, stat='identity',
                   orientation='y', width=0.68, show_legend=False)
        + geom_errorbar(aes(xmin='low', xmax='high'), width=0.18, size=1.0, color='#686868')
        + geom_text(aes(x='label_x', label='label'),
                    data=endpoints[endpoints['method'] != 'BayesianSPO'],
                    hjust=0, size=10, family=PAPER_FONT, color='#333333')
        + geom_text(aes(x='label_x', label='label'),
                    data=endpoints[endpoints['method'] == 'BayesianSPO'],
                    hjust=0, size=10, family=PAPER_FONT, fontface='bold', color='#333333')
        + scale_y_discrete(limits=METHODS[::-1], labels=LABELS[::-1])
        + scale_x_continuous(breaks=[0, 50, 100, 150, 200], limits=[0, 210], expand=[0, 0])
        + labs(x='Final regret (M)', y='', tag='(b)') + style
        + theme(legend_position='none', panel_grid_major_y=element_blank(), axis_ticks_y=element_blank(),
                axis_line_y=element_line(color='#7A7A7A', size=1.0))
    )
    compact = theme(plot_tag_location='plot', plot_tag_position='top',
                    plot_margin=0, plot_inset=[22, 0, 0, 0])
    plot = (
        gggrid([trajectory + compact,
                final_regret + compact + theme(axis_title_y=element_blank(),
                                              plot_inset=[32, 0, 0, 0])],
               ncol=2, widths=[1, 1.35],
               hspace=0, align=True, guides='collect')
        + ggsize(900, 295)
        + theme(plot_margin=0, legend_margin=0,
                legend_text=element_text(family=PAPER_FONT, size=22, color='#000000'),
                legend_key_width=30, legend_key_height=18)
    )
    trajectory_column = (
        trajectory + ggsize(480, 360)
        + guides(color=guide_legend(ncol=3), shape=guide_legend(ncol=3), linetype=guide_legend(ncol=3))
        + theme(plot_tag=element_blank(),
                legend_text=element_text(family=PAPER_FONT, size=16, color='#000000'),
                legend_key_width=24, legend_key_height=16)
    )
    final_regret_column = (
        final_regret + ggsize(560, 260)
        + theme(plot_tag=element_blank(), axis_title_y=element_blank(),
                axis_text_spacing_y=2, plot_margin=[6, 6, 6, 10])
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in [('energy_regret', plot), ('energy_regret_trajectory', trajectory_column),
                         ('energy_regret_histogram', final_regret_column)]:
        svg = Path(ggsave(figure, f'{name}.svg', path=str(output), iframe=False))
        if name != 'energy_regret_trajectory':
            style_bar_svg(svg)
        pdf = output / f'{name}.pdf'
        cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
        subprocess.run(['pdfcrop', '--quiet', '--hires', '--margins', '1', str(pdf), str(pdf)], check=True)
        print(pdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=REPO / 'results/paper/figures/result_figures')
    args = parser.parse_args()
    plot_results(args.source, args.output)
