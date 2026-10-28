"""Nonlinear check: predict each model's deviation from consensus on a confirm-half option from its k nearest explore-half options
(cosine on within-question-centred features), against the linear ridge results in choice_model_fit.md. -- PI/OpenAI, 2026-10-29"""

import numpy as np
import polars as pl

from choice_model_fit import embedding_sets, r2
from normalisation_sweep import EPS, centre, dim_sets, load

qs, dims, models, P, mask, X, src, explore = load()
sets, _ = dim_sets(dims, X, mask, src)
sets |= embedding_sets(qs, mask)
m4 = mask[:, None, None, :]
L = centre(np.log(np.where(m4, P, 1) + EPS), m4)
dev = ((L - L.mean(1, keepdims=True)) * m4).mean(2)  # (Q,M,K)
feats = {'labels (66 virtues + 26 behaviours)': np.concatenate([sets['virt66'], sets['behav26']], -1), 'option+scene embedding': sets['emb_scene']}
rows = []
for source in ('machiavelli', 'daily_dilemmas'):
    tr, te = (src == source) & explore, (src == source) & ~explore
    for fname, F in feats.items():
        Ftr, Fte = F[tr][mask[tr]], F[te][mask[te]]
        n = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        sim = n(Fte) @ n(Ftr).T
        for k in (10, 50, 200):
            nb = np.argsort(-sim, 1)[:, :k]
            for j, mod in enumerate(models):
                ytr, yte = dev[tr, j][mask[tr]], dev[te, j][mask[te]]
                rows.append(dict(source=source, features=fname, k=k, model=mod.split('/')[-1][:8], r2=r2(yte, ytr[nb].mean(1))))
df = pl.DataFrame(rows).group_by('source', 'features', 'k').agg(pl.col('r2').mean().alias('mean_R2_over_models'), pl.col('r2').max().alias('best_model_R2')).sort('source', 'features', 'k')
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(3); pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(40)
print('# kNN held-out R^2 of each model\'s deviation from consensus (confirm half)\n')
print(df)
