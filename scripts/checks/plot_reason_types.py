"""Plot: what each model's written reasoning appeals to (judge_reason_types.py), mean probability per kind with 95% intervals from
resampling questions; games and everyday separately. -- PI/OpenAI, 2026-10-29"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / 'outputs/20261029_reason_types_v12/reason_types.jsonl'
KINDS = [('outcomes', 'Outcomes\n(effects on wellbeing, harm)'), ('character', 'Character\n(honesty, integrity, principles)'),
         ('self_interest', 'Self-interest\n(own benefit, safety, goals)'), ('obligation', 'Obligation\n(rule, duty, promise, role)'),
         ('relationships', 'Relationships\n(care, loyalty to people)'), ('uncertainty', 'Uncertainty\n(avoids acting on unknowns)')]
NAMES = {'claude-haiku-5.5': 'Haiku', 'deepseek-v4.1-flash': 'DeepSeek', 'gemini-3.8-flash': 'Gemini', 'gpt-5.6-luna': 'Luna',
         'muse-spark-1.3-contributor': 'Muse', 'qwen3.8-27b': 'Qwen'}
COLORS = {'Muse': '#7b3294', 'Qwen': '#1b9e77', 'Gemini': '#c65a00', 'Luna': '#007a92', 'DeepSeek': '#555555', 'Haiku': '#d6336c'}
rows = [json.loads(x) for x in J.open()]
src = {q: ('games' if q.startswith('machiavelli') else 'everyday') for q in {r['source_id'] for r in rows}}
rng = np.random.default_rng(20261029)
export = {'kinds': [k for k, _ in KINDS], 'labels': [l.replace('\n', ' ') for _, l in KINDS], 'sources': {}}
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
for ax, (s, title) in zip(axs, (('games', 'Game scenes'), ('everyday', 'Everyday dilemmas'))):
    for j, (model, name) in enumerate(NAMES.items()):
        mine = [r for r in rows if src[r['source_id']] == s and r['model'].endswith(model)]
        qids = sorted({r['source_id'] for r in mine})
        per_q = {q: [] for q in qids}
        for r in mine:
            per_q[r['source_id']].append([r['probabilities'][k] for k, _ in KINDS])
        A = np.array([np.mean(per_q[q], 0) for q in qids])  # questions x kinds (mean of the 2 answers)
        boot = np.array([A[rng.integers(0, len(A), len(A))].mean(0) for _ in range(1000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], 0)
        export['sources'].setdefault(s, {})[name] = {'mean': A.mean(0).round(4).tolist(), 'lo': lo.round(4).tolist(), 'hi': hi.round(4).tolist(), 'questions': len(A)}
        y = np.arange(len(KINDS)) + (j - 2.5) * .11
        ax.errorbar(A.mean(0), y, xerr=[A.mean(0) - lo, hi - A.mean(0)], fmt='o', ms=5, color=COLORS[name], elinewidth=1, capsize=0, label=name)
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_xlim(0, .6)
    ax.set_xlabel('share of reasoning that appeals to this (judged)', fontsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=.25)
    ax.text(.5, .02, 'wassname/virtue-choices', transform=ax.transAxes, ha='center', fontsize=7, style='italic', color='#cccccc')
axs[0].set_yticks(range(len(KINDS)), [k[1] for k in KINDS], fontsize=8)
axs[0].invert_yaxis()
axs[1].legend(fontsize=8, frameon=False, loc='lower right')
(J.parent / 'reason_types_site.json').write_text(json.dumps(export, indent=1))
out = ROOT / 'outputs/20261029_reason_types_v12/reason_types.png'
fig.savefig(out, dpi=130)
print(out)
