"""Abliterated DeepSeek (OrcaCyber, 2 answers) and the persona controls (1 answer), against the v12 models. Based on the positive control: DeepSeek with a stated value system (utilitarian, rule-follower, selfish, loyal), 1 answer per question, compared
with the six v12 models. Same per-question statistic (cov of sharpness-removed log p with centred labels), averaged per source; all
shown minus DeepSeek v12 (mean of its 2 answers). Label-free: Jensen-Shannon to DeepSeek v12 answer 1. -- PI/OpenAI, 2026-10-29"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from normalisation_sweep import dim_sets, load, p_transforms, stat

ROOT = Path(__file__).resolve().parents[2]
PERSONAS = ['utilitarian', 'rule-follower', 'selfish', 'loyal']
ORCA = ROOT / 'outputs/20261029_random1000_v12_orca'
qs, dims, models, P, mask, X, src, explore = load()
sets, groups = dim_sets(dims, X, mask, src)
qi = {q['source_id']: i for i, q in enumerate(qs)}
Pp = np.full((len(qs), len(PERSONAS), 1, P.shape[-1]), np.nan)
for j, k in enumerate(PERSONAS):
    for f in (ROOT / f'outputs/20261029_persona_{k}').glob('*/choices.jsonl'):
        for r in map(json.loads, f.open()):
            if r['status'] == 'ok' and r['source_id'] in qi:
                a = qi[r['source_id']]
                Pp[a, j, 0, :int(mask[a].sum())] = [r['original_probabilities'][o] / 100 for o in qs[a]['options']]
Po = np.full((len(qs), 1, 2, P.shape[-1]), np.nan)
for f in ORCA.glob('*/choices.jsonl'):
    for r in map(json.loads, f.open()):
        if r['status'] == 'ok' and r['source_id'] in qi:
            a = qi[r['source_id']]
            Po[a, 0, r['draw'], :int(mask[a].sum())] = [r['original_probabilities'][o] / 100 for o in qs[a]['options']]
ok = np.isfinite(Pp[:, :, 0, 0]).all(1) & np.isfinite(Po[:, 0, :, 0]).all(1)  # questions every persona and both abliterated answers cover
print(f'{ok.sum()} questions with all four personas')
ds = models.index('deepseek/deepseek-v4.1-flash')
allP = np.concatenate([P, np.nan_to_num(Po), np.repeat(np.nan_to_num(Pp), 2, 2)], 1)  # personas duplicated into both answer slots
names = [m.split('/')[-1][:10] for m in models] + ['DS abliterated'] + ['DS+' + k for k in PERSONAS]
T = p_transforms(allP, mask)['logp_z']


def js(a, b):
    a, b = a + 5e-3, b + 5e-3; a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True); m = (a + b) / 2
    return .5 * (a * np.log(a / m)).sum(-1) + .5 * (b * np.log(b / m)).sum(-1)


rows = []
for source in ('machiavelli', 'daily_dilemmas'):
    sel = (src == source) & ok
    row_js = {}
    for j, n in enumerate(names):
        d = [js(allP[a, j, 0, :int(mask[a].sum())], P[a, ds, 0, :int(mask[a].sum())]) for a in np.flatnonzero(sel)]
        row_js[n] = np.mean(d)
    noise = np.mean([js(P[a, ds, 1, :int(mask[a].sum())], P[a, ds, 0, :int(mask[a].sum())]) for a in np.flatnonzero(sel)])
    for setname in ('dnd', 'groups9'):
        S = np.nanmean(stat(T, sets[setname], mask, 'cov'), 2)  # (Q, M+4, D)
        prof = np.nanmean(S[sel], 0)
        prof -= prof[ds]
        labels = ['lawful', 'good'] if setname == 'dnd' else list(groups)
        for j, n in enumerate(names):
            if j == ds:
                continue
            rows.append(dict(source=source, labels=setname, who=n, JS_to_DeepSeek=row_js[n] - noise, **{l: prof[j, i] for i, l in enumerate(labels)}))
df = pl.DataFrame(rows, strict=False)
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(3); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
print('# Positive control: persona DeepSeek vs the v12 models, each minus DeepSeek v12 (sharpness removed)\n')
print('JS_to_DeepSeek: label-free distance of single answers to DeepSeek answer 1, minus DeepSeek answer 2 vs answer 1 (0 = like DeepSeek repeated)\n')
for s in ('dnd', 'groups9'):
    print(df.filter(pl.col('labels') == s).drop([c for c in df.columns if df.filter(pl.col('labels') == s)[c].null_count() == len(df.filter(pl.col('labels') == s))]))
    print()
