"""One table of aggregation variants for separating models, all on calibrated per-answer log p (the published method) unless named.
Each variant gives S[q,m,r,d]; separation() reports confirm-half split-half reliability (rel_q), model identification (id_q, chance
1/6) and size (effect = between-model sd / between-question sd). Shuffled-model null is ~0 / 0.17. Variants:
  weights:   equal | game cluster (each cluster equal) | contested (1 - consensus top) | disagreement (between-model JS of the question)
  ambiguity: questions split into tertiles of consensus entropy, separation within each
  per-scene common factor: consensus removed then (a) rank-1 per scene (PCA of the models x options deviation matrix, top component),
             (b) per-scene normalised (deviation / its RMS, so each scene's disagreement counts equally),
             (c) options weighted by between-model sd (contested options count more)
  probabilities: raw log p, sharpness removed (z), calibrated
-- PI/OpenAI, 2026-10-29"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from normalisation_sweep import centre, dim_sets, load, p_transforms, separation, stat  # noqa: E402
from plot_probability_eval import calibrated_logp, load_calibration  # noqa: E402

qs, dims, models, P, mask, X, src, explore = load()
sets, _ = dim_sets(dims, X, mask, src)
_, maps, _, fold = load_calibration(ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
m4 = mask[:, None, None, :]
C = np.zeros_like(P)
for a, q in enumerate(qs):
    for b, m in enumerate(models):
        for r in (0, 1):
            C[a, b, r, :len(q['options'])] = calibrated_logp(q, m, maps, fold, (r,))
T = {'raw log p': p_transforms(P, mask)['logp'], 'sharpness removed': p_transforms(P, mask)['logp_z'], 'calibrated': centre(C, m4)}
cal = T['calibrated']
dev = (cal - cal.mean(1, keepdims=True)) * m4  # consensus removed (same answer index)

# per-scene common-factor variants
rank1 = np.zeros_like(dev)
for a in range(len(qs)):
    k = int(mask[a].sum())
    for r in (0, 1):
        U, s, Vt = np.linalg.svd(dev[a, :, r, :k], full_matrices=False)
        rank1[a, :, r, :k] = s[0] * np.outer(U[:, 0], Vt[0])
rms = np.sqrt((dev ** 2).sum((1, 3), keepdims=True) / (mask.sum(-1)[:, None, None, None] * len(models)))
scene_norm = np.divide(dev, rms, out=np.zeros_like(dev), where=rms > 1e-9)
opt_sd = cal.mean(2).std(1)[:, None, None, :] * m4  # between-model sd of each option (Q,1,1,K)
opt_w = opt_sd / np.maximum(opt_sd.sum(-1, keepdims=True), 1e-9) * mask.sum(-1)[:, None, None, None]
variants = {'calibrated (baseline)': cal, 'raw log p': T['raw log p'], 'sharpness removed (z)': T['sharpness removed'],
            'consensus removed': dev, 'per-scene rank-1 (PCA per scene)': rank1, 'per-scene normalised': scene_norm,
            'options weighted by between-model sd': dev * opt_w}

pbar = P.mean((1, 2))
ent = -(np.where(mask, pbar, 1) * np.log(np.where(mask, pbar, 1))).sum(-1) / np.log(mask.sum(-1))
jsq = np.array([np.mean([0.5 * ((P[a, i, :, :k] * np.log((P[a, i, :, :k] + 1e-3) / ((P[a, i, :, :k] + P[a, j, :, :k]) / 2 + 1e-3))).sum(-1)
                                + (P[a, j, :, :k] * np.log((P[a, j, :, :k] + 1e-3) / ((P[a, i, :, :k] + P[a, j, :, :k]) / 2 + 1e-3))).sum(-1)).mean()
                         for i in range(len(models)) for j in range(i + 1, len(models))]) for a, k in enumerate(mask.sum(-1))])
cluster = np.array([q['eval_cluster'] for q in qs])
cw = np.array([1 / (cluster == c).sum() for c in cluster])
weights = {'equal': np.ones(len(qs)), 'game cluster': cw, 'contested': 1 - pbar.max(-1), 'disagreement (JS)': jsq}
def kendall(Tv, Xs):
    # per question: mean over option pairs of sign(l_i - l_j) * sign(x_i - x_j); ties count 0; questions with no label difference -> NaN
    K = mask.shape[1]
    iu = np.triu_indices(K, 1)
    pm = (mask[:, iu[0]] & mask[:, iu[1]]).astype(float)  # (Q,pairs)
    sl = np.sign(Tv[..., iu[0]] - Tv[..., iu[1]]) * pm[:, None, None, :]  # (Q,M,R,pairs)
    sx = np.sign(Xs[:, iu[0], :] - Xs[:, iu[1], :]) * pm[..., None]  # (Q,pairs,D)
    out = np.einsum('qmrp,qpd->qmrd', sl, sx) / np.maximum(pm.sum(-1), 1)[:, None, None, None]
    return np.where((np.abs(sx).sum(1) > 0)[:, None, None, :], out, np.nan)


rng3 = np.random.default_rng(3)
rows = []
for name in ('dnd', 'groups9', 'virt66', 'style9'):
    Xs = sets[name]
    for source in ('machiavelli', 'daily_dilemmas'):
        sel = (src == source) & ~explore
        def add(family, variant, S, w=weights['equal'], s=sel):
            rows.append(dict(source=source, labels=name, family=family, variant=variant, questions=int(s.sum()), **separation(S, w, s)))
        for v, Tv in variants.items():
            add('probabilities / per-scene', v, stat(Tv, Xs, mask, 'cov'))
            if v not in ('calibrated (baseline)', 'raw log p', 'sharpness removed (z)'):
                rows[-1]['effect'] = float('nan')  # consensus removed: per-question model mean is 0, so between-question sd is undefined
        add('pairing statistic', "Kendall tau (pair signs, ordinal both sides)", kendall(T['raw log p'], Xs))
        S = stat(cal, Xs, mask, 'cov')
        for w, wv in weights.items():
            if w != 'equal':
                add('question weights', w, S, wv)
        tert = np.quantile(ent[src == source], [1 / 3, 2 / 3])
        for b, (lo, hi) in enumerate(((-1, tert[0]), (tert[0], tert[1]), (tert[1], 2))):
            add('ambiguity tertile', ['clear (low entropy)', 'middle', 'ambiguous (high entropy)'][b], S, s=sel & (ent > lo) & (ent <= hi))
        thirds = [separation(S, weights['equal'], sel & (rng3.random(len(qs)) < 1 / 3)) for _ in range(10)]
        rows.append(dict(source=source, labels=name, family='ambiguity tertile', variant='random third (control, mean of 10)', questions=int(sel.sum() / 3),
                         **{k: float(np.nanmean([t[k] for t in thirds])) for k in thirds[0]}))
df = pl.DataFrame(rows).drop('rel_r')
df.write_parquet(ROOT / 'outputs/20261027_random1000_v12/aggregation_sweep.parquet')
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_rows(200); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_cols(16)
print('# Aggregation sweep (confirm half). rel_q: split-half reliability (null ~0); id_q: model identification (chance 0.17); effect: between-model / between-question sd\n')
wide = df.pivot(on='source', index=['labels', 'family', 'variant'], values=['rel_q', 'id_q', 'effect'])
wide = df.pivot(on='source', index=['labels', 'family', 'variant'], values=['rel_q', 'id_q', 'effect', 'spread'])
print(wide.select('labels', 'family', 'variant', 'rel_q_machiavelli', 'id_q_machiavelli', 'effect_machiavelli', 'spread_machiavelli',
                  'rel_q_daily_dilemmas', 'id_q_daily_dilemmas', 'effect_daily_dilemmas', 'spread_daily_dilemmas'))

# label-free: single-answer JS between models vs within a model, by ambiguity tertile
def jsd(a, b, k):
    a, b = a[:k] + 5e-3, b[:k] + 5e-3; a, b = a / a.sum(), b / b.sum(); m = (a + b) / 2
    return .5 * (a * np.log(a / m)).sum() + .5 * (b * np.log(b / m)).sum()
lf = []
for source in ('machiavelli', 'daily_dilemmas'):
    tert = np.quantile(ent[src == source], [1 / 3, 2 / 3])
    for b, (lo, hi) in enumerate(((-1, 2), (-1, tert[0]), (tert[0], tert[1]), (tert[1], 2))):
        qsel = np.flatnonzero((src == source) & (ent > lo) & (ent <= hi))
        btw = np.mean([jsd(P[a, i, r], P[a, j, t], int(mask[a].sum())) for a in qsel for i in range(6) for j in range(i + 1, 6) for r in (0, 1) for t in (0, 1)])
        wit = np.mean([jsd(P[a, i, 0], P[a, i, 1], int(mask[a].sum())) for a in qsel for i in range(6)])
        lf.append(dict(source=source, questions=['all', 'clear (low entropy)', 'middle', 'ambiguous (high entropy)'][b], n=len(qsel), JS_between=btw, JS_within=wit, excess=btw - wit, ratio=btw / wit))
print('\n## Label-free separation (single answers, Jensen-Shannon, nats)\n')
pl.Config.set_float_precision(3)
print(pl.DataFrame(lf))
