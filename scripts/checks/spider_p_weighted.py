"""Figure 4b (kinds of good, goodness removed, vs DeepSeek) with pairs weighted by label gap only (as on the page)
versus label gap x p_i p_j (how likely both options are to be chosen, pooled over the other models, leave-one-out).
Question: are models samey because obvious pairs dominate? Compare between-model spread with DeepSeek-vs-itself. -- PI/OpenAI"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from plot_probability_eval import ROOT, SOURCES, calibrated_logp, load_calibration, short, virtue_basis  # noqa: E402

tables = ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl'
lines, maps, identity, fold = load_calibration(tables)
groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
names = list(groups)
basis = virtue_basis(lines, groups)
gidx = [[basis['ids'].index(i) for i in items] for items in groups.values()]
ds = 'deepseek/deepseek-v4.1-flash'
models = [m for m in lines[1]['models'] if m != ds]

for src, sname in SOURCES.items():
    v1 = basis['src'][src]['V'][0]
    acc = {k: {m: np.zeros((2, len(names))) for m in models + ['self']} for k in ('gap', 'p_weighted', 'contested10', 'contested20', 'contested10_p')}
    for q in lines[1:]:
        if q['source'] != src or set(models + [ds]) - set(q['models']) or any(len(q['models'][m]['draws']) < 2 for m in q['models']):
            continue
        o = list(q['options'])
        X = np.array([q['option_scores'][x] for x in o], float)[:, basis['direct']] * basis['fav']
        if not np.isfinite(X).all():
            continue
        R = X - np.outer(X @ v1, v1)
        L = np.stack([R[:, ix].mean(1) for ix in gidx], 1)
        iu = np.triu_indices(len(o), 1)
        dL = L[iu[0]] - L[iu[1]]
        pct = {m: np.array([q['models'][m]['mean_stated_percent'][x] for x in o]) / 100 for m in q['models']}
        ref = calibrated_logp(q, ds, maps, fold)
        d0, d1 = [np.log(np.array([dr['probabilities'][x] for x in o], float) + 1) for dr in q['models'][ds]['draws'][:2]]
        for m in models + ['self']:
            delta = (d0 - d1) if m == 'self' else calibrated_logp(q, m, maps, fold) - ref
            dd = (delta[iu[0]] - delta[iu[1]])[:, None]
            others = np.mean([pct[k] for k in q['models'] if k != m], 0)  # leave the scored model out of the weights
            pp = (others[iu[0]] * others[iu[1]])[:, None]
            live = lambda c: ((others[iu[0]] >= c) & (others[iu[1]] >= c))[:, None]  # both options picked at least c by the other models
            for k, w in (('gap', np.abs(dL)), ('p_weighted', np.abs(dL) * pp), ('contested10', np.abs(dL) * live(.1)), ('contested20', np.abs(dL) * live(.2)), ('contested10_p', np.abs(dL) * live(.1) * pp)):
                acc[k][m][0] += (w * np.sign(dL) * dd).sum(0)
                acc[k][m][1] += w.sum(0)
    print(f'\n## {sname}')
    for k in ('gap', 'p_weighted', 'contested10', 'contested20', 'contested10_p'):
        print(f'\n{k}: nats vs DeepSeek per virtue group (goodness removed)')
        print(f"{'':20s}" + ''.join(f'{g[:9]:>10s}' for g in names))
        vals = {}
        for m in models + ['self']:
            v = acc[k][m][0] / acc[k][m][1]
            vals[m] = v
            print(f"{'DeepSeek vs itself' if m == 'self' else short(m):20s}" + ''.join(f'{x:+10.3f}' for x in v))
        between = np.std([vals[m] for m in models], 0).mean()
        print(f'mean between-model sd {between:.3f}; mean |DeepSeek vs itself| {np.abs(vals["self"]).mean():.3f}; ratio {between / np.abs(vals["self"]).mean():.1f}')
