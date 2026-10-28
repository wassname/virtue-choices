"""Outcome-visible pilot (48 game scenes): compare (1) labels judged with vs without the outcome, (2) model choices with vs without the
outcome shown, under each label set. Uncalibrated and descriptive (pilot): expected label of the chosen action = sum_i p_i x_i, model
minus DeepSeek; spread between models; mean pairwise total-variation (TV) distance between models' choice distributions. -- PI/OpenAI"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V12 = ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl'
RUN = ROOT / 'outputs/20261028_outcome_pilot'
LAB = ROOT / 'outputs/20261028_outcome_labels'
DS = 'deepseek/deepseek-v4.1-flash'

lines = [json.loads(x) for x in V12.open()]
dims = lines[0]['_meta']['dimensions']
scenes = {json.loads(x)['source_id']: json.loads(x) for x in (ROOT / 'outputs/20261028_outcome_pilot_selection/scenes.jsonl').open()}
qs = {q['source_id']: q for q in lines[1:] if q['source_id'] in scenes}
idx = {d['id']: i for i, d in enumerate(dims)}
virt = [i for i, d in enumerate(dims) if d['family'] == 'virtues' and d['derived_from'] is None]
sign = {i: (-1. if dims[i]['kind'] == 'vices' else 1.) for i in virt}

# new labels: (source_id, choice_id) -> {dim id: score}
new = defaultdict(dict)
for kind, key in (('virtues', 'item_id'), ('dnd', 'axis')):
    for r in map(json.loads, (LAB / kind / 'labels.jsonl').open()):
        if r.get('derived_from') or r.get('derived'):
            continue
        score = r.get('signed', r.get('score'))
        new[r['source_id'], r['choice_id']][r[key]] = score

# outcome-visible answers: mean stated % per (scene, model) over valid draws, original option ids
ans = defaultdict(lambda: defaultdict(list))
for st in ('smoke', 'rest'):
    for r in map(json.loads, (RUN / st / 'choices.jsonl').open()):
        if r['status'] == 'ok':
            ans[r['source_id']][r['model']].append(r['original_probabilities'])
after = {s: {m: {k: np.mean([d[k] for d in v]) / 100 for k in v[0]} for m, v in ms.items()} for s, ms in ans.items()}
before = {s: {m: {k: v / 100 for k, v in q['models'][m]['mean_stated_percent'].items()} for m in q['models']} for s, q in qs.items()}
models = sorted({m for s in after for m in after[s]})

print('# Outcome-visible pilot, 48 game scenes\n')
print('## 1. Labels: judged with the outcome visible vs the original visible-only labels\n')
rows = []
for s, q in qs.items():
    for k in q['options']:
        old = q['option_scores'][k]
        n = new[s, k]
        for name in ('order', 'morality'):
            if old[idx[name]] is not None and n.get(name) is not None:
                rows.append(('dnd_' + name, old[idx[name]], n[name], s, k))
        for i in virt:
            if dims[i]['id'] in n and old[i] is not None and n[dims[i]['id']] is not None:
                rows.append(('virtue', old[i] * sign[i], n[dims[i]['id']], s, k))  # stored scores are 'more of the named trait'; new 'signed' is already favourable
for kind in ('dnd_order', 'dnd_morality', 'virtue'):
    a = np.array([(o, nn) for kk, o, nn, *_ in rows if kk == kind], float)
    a = a[np.isfinite(a).all(1)]  # D&D can be null (unclear)
    print(f'{kind:12s} n={len(a):5d}  corr old-new {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.2f}  mean |change| {np.abs(a[:, 1] - a[:, 0]).mean():.2f}  mean change {np.mean(a[:, 1] - a[:, 0]):+.2f}')
print('\nLargest D&D good (morality) changes:')
dm = sorted([r for r in rows if r[0] == 'dnd_morality' and np.isfinite([r[1], r[2]]).all()], key=lambda r: -abs(r[2] - r[1]))[:8]
for _, o, nn, s, k in dm:
    print(f'  {o:+.2f} -> {nn:+.2f}  {qs[s]["options"][k][:70]!r}  | outcome: {scenes[s]["outcomes"][k][:110]!r}')


def label_vec(s, k, which, dim):
    if which == 'old':
        v = qs[s]['option_scores'][k][idx[dim]]
    else:
        v = new[s, k].get(dim)
    return np.nan if v is None else v


def expected(prob, s, which, dim):
    xs = np.array([label_vec(s, k, which, dim) for k in qs[s]['options']])
    ps = np.array([prob[k] for k in qs[s]['options']])
    ok = np.isfinite(xs)
    return np.nan if ok.sum() < 2 else float((ps[ok] / ps[ok].sum() * xs[ok]).sum() - xs[ok].mean())


print('\n## 2. Choices: D&D good of the chosen action (expected label minus the scene mean), model minus DeepSeek\n')
print(f"{'':10s}" + ''.join(f'{x:>22s}' for x in ('no outcome, old labels', 'outcome, old labels', 'outcome, new labels')))
spreads = []
for m in models:
    if m == DS:
        continue
    vals = []
    for prob_set, which in ((before, 'old'), (after, 'old'), (after, 'new')):
        d = [expected(prob_set[s][m], s, which, 'morality') - expected(prob_set[s][DS], s, which, 'morality') for s in qs if m in prob_set[s] and DS in prob_set[s]]
        vals.append(np.nanmean(d))
    spreads.append(vals)
    print(f"{m.split('/')[-1][:10]:10s}" + ''.join(f'{v:+22.3f}' for v in vals))
print(f"{'sd':10s}" + ''.join(f'{v:22.3f}' for v in np.std(spreads, 0)))


def tv(a, b):
    return 0.5 * sum(abs(a[k] - b[k]) for k in a)


print('\n## 3. How differently the models answer (mean pairwise TV between models)\n')
for name, ps in (('no outcome (v12)', before), ('outcome shown', after)):
    t = [tv(ps[s][a], ps[s][b]) for s in qs for i, a in enumerate(models) for b in models[i + 1:] if a in ps[s] and b in ps[s]]
    top = [max(ps[s][a], key=ps[s][a].get) == max(ps[s][b], key=ps[s][b].get) for s in qs for i, a in enumerate(models) for b in models[i + 1:] if a in ps[s] and b in ps[s]]
    print(f'{name:18s} mean TV {np.mean(t):.3f}   same top action {np.mean(top):.0%}')

print('\n## 4. Uncertainty: paired bootstrap over scenes (2,000 resamples), outcome shown minus no outcome\n')
rng = np.random.default_rng(20261028)
S = [s for s in qs if all(m in before[s] and m in after[s] for m in models)]
pair_tv = lambda ps, s: np.mean([tv(ps[s][a], ps[s][b]) for i, a in enumerate(models) for b in models[i + 1:]])
d_tv = np.array([pair_tv(after, s) - pair_tv(before, s) for s in S])
boot = [d_tv[rng.integers(0, len(S), len(S))].mean() for _ in range(2000)]
print(f'mean TV between models: change {d_tv.mean():+.3f}, 95% [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}] ({len(S)} scenes with all six models)')
ds_self = {}
for name, path in (('outcome shown', RUN),):
    pass
dsr = defaultdict(list)
for st in ('smoke', 'rest'):
    for r in map(json.loads, (RUN / st / 'choices.jsonl').open()):
        if r['status'] == 'ok' and r['model'] == DS:
            dsr[r['source_id']].append({k: v / 100 for k, v in r['original_probabilities'].items()})
print(f"DeepSeek answer 1 vs answer 2 (noise floor), outcome shown: mean TV {np.mean([tv(v[0], v[1]) for v in dsr.values() if len(v) == 2]):.3f}")
