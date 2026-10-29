"""Do within-question (comparative) labels separate models better than absolute per-option labels? Labels: q[d,i] = judge's
probability that option i shows dimension d most (label_comparative.py); feature x = log q centred within the question.
(1) separation: same cov statistic and confirm-half reliability / ID as aggregation_sweep (calibrated log p).
(2) fit of each dimension's ranking to each model's ranking: mean Kendall tau over questions (pair signs, ties 0).
(3) preference model: per model, ridge of its deviation from consensus (double-centred log p) on the 12 log q features,
    fitted on explore and scored on confirm, against the same fit with the 66 absolute virtue labels. -- PI/OpenAI, 2026-10-29"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from choice_model_fit import LAMBDAS, inner_cv, r2  # noqa: E402
from normalisation_sweep import EPS, centre, dim_sets, load, separation, stat  # noqa: E402
from plot_probability_eval import calibrated_logp, load_calibration  # noqa: E402

qs, dims, models, P, mask, X, src, explore = load()
sets, _ = dim_sets(dims, X, mask, src)
qi = {q['source_id']: i for i, q in enumerate(qs)}
rows = [json.loads(x) for x in (ROOT / 'outputs/20261029_comparative_labels/comparative_labels.jsonl').open()]
D = list(dict.fromkeys(r['dim'] for r in rows))
Qd = np.full((len(qs), mask.shape[1], len(D)), np.nan)
for r in rows:
    if r['source_id'] not in qi:  # the one question Haiku never answered validly (load() keeps complete questions only)
        continue
    a = qi[r['source_id']]
    for j, o in enumerate(qs[a]['options']):
        Qd[a, j, D.index(r['dim'])] = r['probabilities'][o]
have = np.isfinite(Qd[:, 0, :]).all(-1)
F = centre(np.log(np.where(mask[..., None], np.nan_to_num(Qd), 1) + .01), mask[..., None], axis_k=1)
F[~have] = 0
_, maps, _, fold = load_calibration(ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
m4 = mask[:, None, None, :]
C = np.zeros_like(P)
for a, q in enumerate(qs):
    for b, m in enumerate(models):
        for r in (0, 1):
            C[a, b, r, :len(q['options'])] = calibrated_logp(q, m, maps, fold, (r,))
C = centre(C, m4)
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
print(f'# Comparative labels: {have.sum()} questions x {len(D)} dimensions\n')

out = []
for source in ('machiavelli', 'daily_dilemmas'):
    sel = (src == source) & ~explore & have
    for name, Xs in (('comparative 12 dims', F), ('comparative 9 groups', F[..., :9]), ('absolute 9 groups', sets['groups9']), ('absolute D&D', sets['dnd'])):
        sep = separation(stat(C, Xs, mask, 'cov'), np.ones(len(qs)), sel)
        out.append(dict(source=source, labels=name, questions=int(sel.sum()), rel_q=sep['rel_q'], id_q=sep['id_q']))
print('## (1) Separation, confirm half, calibrated log p (same metric as aggregation_sweep)\n')
print(pl.DataFrame(out).pivot(on='source', index='labels', values=['rel_q', 'id_q']))

iu = {k: np.triu_indices(k, 1) for k in range(2, mask.shape[1] + 1)}
tau = np.full((len(qs), len(models), len(D)), np.nan)
for a in np.flatnonzero(have):
    k = int(mask[a].sum()); i, j = iu[k]
    sx = np.sign(Qd[a, i] - Qd[a, j])  # (pairs, D)
    sp = np.sign(P[a][:, :, i] - P[a][:, :, j]).mean(1)  # (M, pairs), mean of 2 answers
    tau[a] = sp @ sx / len(i)
print('\n## (2) Kendall tau between each model\'s ranking and each dimension\'s ranking (mean over questions; 1 = always agrees)\n')
for source in ('machiavelli', 'daily_dilemmas'):
    t = np.nanmean(tau[(src == source) & have], 0)
    df = pl.DataFrame({'model': [m.split('/')[-1][:10] for m in models], **{d: t[:, j] for j, d in enumerate(D)}})
    print(f'### {source}\n'); print(df)
    print('spread across models (sd): ' + ', '.join(f'{d} {t[:, j].std():.3f}' for j, d in enumerate(D)) + '\n')

dev = ((C - C.mean(1, keepdims=True)) * m4).mean(2)
fit = []
for source in ('machiavelli', 'daily_dilemmas'):
    tr, te = (src == source) & explore & have, (src == source) & ~explore & have
    for name, Xs in (('comparative 12 dims', F), ('absolute 66 virtues', sets['virt66'])):
        Xtr, Xte = Xs[tr][mask[tr]], Xs[te][mask[te]]
        for j, mod in enumerate(models):
            ytr, yte = dev[tr, j][mask[tr]], dev[te, j][mask[te]]
            lam = max(LAMBDAS, key=lambda l: inner_cv(Xtr, ytr, l, tr, mask[tr]))
            beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
            fit.append(dict(source=source, labels=name, model=mod.split('/')[-1][:10], r2=r2(yte, Xte @ beta)))
print('\n## (3) Preference model: held-out R^2 of each model\'s deviation from consensus (calibrated)\n')
print(pl.DataFrame(fit).pivot(on='labels', index=['source', 'model'], values='r2'))
