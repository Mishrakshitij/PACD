#!/usr/bin/env python3
"""Plot measured counts from the committed dataset manifest."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'datasets/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'docs/figures')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Dataset release must be complete')
    rows = manifest['files']
    names = ['DSTC1' if row['domain'] == 'dstc1' else row['domain'].title() for row in rows]
    colors = ['#64748b', '#819bbe', '#72b7ae', '#0f766e']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'svg.fonttype': 'none', 'svg.hashsalt': 'pacd-release'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1, 1.4]})
    counts = [row['rows'] for row in rows]
    bars = axes[0].bar(names, counts, color='#0f766e', width=.65)
    axes[0].bar_label(bars, labels=[f'{count:,}' for count in counts], padding=4, fontsize=10)
    axes[0].set_ylim(0, max(counts) * 1.15)
    axes[0].set_ylabel('Utterances')
    axes[0].set_title('Released annotations', pad=14)
    left = [0.] * len(rows)
    for label, color in enumerate(colors):
        shares = [100 * row['label_counts'][str(label)] / row['rows'] for row in rows]
        bars = axes[1].barh(names, shares, left=left, color=color,
                            label=manifest['label_order'][label].replace('_', ' ').capitalize(), height=.65)
        for bar, share in zip(bars, shares):
            if share > 7:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                             f'{share:.1f}%', ha='center', va='center', color='white', fontsize=9)
        left = [a + b for a, b in zip(left, shares)]
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Share of released annotations (%)')
    axes[1].set_title('Measured politeness distribution', pad=14)
    fig.suptitle('PACD annotation release', fontsize=18, fontweight='bold', color='#1e293b', y=.98)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .03), ncol=4, frameon=False)
    fig.text(.5, .01, f"Measured from release {manifest['release']} · Separate from the manuscript's gold annotations",
             ha='center', color='#64748b', fontsize=9)
    fig.tight_layout(rect=[0, .15, 1, .9])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        path = args.output_dir / f'dataset_release.{suffix}'
        fig.savefig(path, dpi=200, bbox_inches='tight', metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)


if __name__ == '__main__':
    main()
