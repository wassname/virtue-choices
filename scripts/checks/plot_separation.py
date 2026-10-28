"""Figure: how reliably each measurement tells the six models apart (split-half reliability on held-out questions), with and without
each model's sharpness (how extreme its stated percentages are). Inputs: normalisation_sweep.parquet, reason_type_profiles.json. -- PI/OpenAI"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/20261027_random1000_v12'
ROWS = [('goodness', 'Goodness (1 axis)'), ('dnd', 'D&D (2 axes)'), ('groups9', '9 virtue groups'), ('virt66', '66 virtues'),
        ('behav26', '26 behaviours'), ('style9', 'Style: 9 groups,\ngoodness removed'), ('reasons', 'Kind of reason given\n(8 kinds, judged)')]
df = pl.read_parquet(RUN / 'normalisation_sweep.parquet').filter(
    (pl.col('half') == 'confirm') & (pl.col('labels') == 'raw') & (pl.col('stat') == 'cov'))
reasons = json.loads((ROOT / 'outputs/20261029_reason_types_v12/reason_type_profiles.json').read_text())
fig, axs = plt.subplots(2, 2, figsize=(9, 6.4), sharey=True, constrained_layout=True)
for col, (src, title) in enumerate((('machiavelli', 'Game scenes'), ('daily_dilemmas', 'Everyday dilemmas'))):
    for row, (metric, xlabel, xlim) in enumerate((('rel_q', 'split-half reliability of model differences', (-.2, 1.02)),
                                                   ('effect', 'size: spread between models / spread between questions', (0, .25)))):
        ax = axs[row, col]
        if metric == 'rel_q':
            ax.axvspan(-.15, .15, color='#eeeeee', zorder=0)
            ax.text(0, 3, 'shuffled\nmodels', ha='center', va='center', fontsize=7, color='#999999')
        for y, (key, label) in enumerate(ROWS):
            if key == 'reasons':
                ax.plot(reasons[src]['separation']['confirm'][metric], y, 'o', color='#7b3294', ms=7)
                continue
            get = lambda probs: df.filter((pl.col('source') == src) & (pl.col('dims') == key) & (pl.col('probs') == probs) & (pl.col('weight') == 'equal'))[metric][0]
            a, b = get('logp'), get('logp_z')
            ax.plot([a, b], [y, y], color='#bbbbbb', lw=1, zorder=1)
            ax.plot(a, y, 'o', mfc='white', mec='#333333', ms=6, zorder=2)
            ax.plot(b, y, 'o', color='#333333', ms=6, zorder=3)
        if row == 0:
            ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlim(*xlim)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.text(.5, .02, 'wassname/virtue-choices', transform=ax.transAxes, ha='center', fontsize=7, style='italic', color='#cccccc')
for ax in axs[:, 0]:
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=8)
axs[0, 0].invert_yaxis()
axs[0, 1].text(1.0, -.9, '○ labels, raw   ● labels, sharpness removed   purple: reasons', ha='right', fontsize=7, color='#333333')
out = RUN / 'separation_by_measurement.png'
fig.savefig(out, dpi=130)
print(out)
