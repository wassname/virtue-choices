"""Do models differ more once goodness cannot drive the comparison? Pairwise within-question statistic
    S[q,m,r,d] = mean over selected option pairs (i,j) of (l_i - l_j) * (x_i - x_j)       l = log p scaled to unit sd over options (sharpness removed)
with pairs selected as: all; matched (|goodness_i - goodness_j| below the 25th percentile of pair gaps in that source); good-only
(both options D&D good > 1). Same split-half reliability / identification / effect metrics as normalisation_sweep. -- PI/OpenAI, 2026-10-29"""

import itertools

import numpy as np
import polars as pl

from normalisation_sweep import dim_sets, load, p_transforms, separation

qs, dims, models, P, mask, X, src, explore = load()
sets, _ = dim_sets(dims, X, mask, src)
L = p_transforms(P, mask)['logp_z']
good = sets['goodness'][..., 0]
dnd_good = X[:, :, [d['id'] for d in dims].index('morality')]
pairs = [(q, i, j) for q in range(len(qs)) for i, j in itertools.combinations(range(int(mask[q].sum())), 2)]
gap = np.array([abs(good[q, i] - good[q, j]) for q, i, j in pairs])
srcp = np.array([src[q] for q, _, _ in pairs])
thr = {s: np.percentile(gap[srcp == s], 25) for s in np.unique(src)}
rules = {'all pairs': lambda q, i, j, g: True,
         'matched goodness': lambda q, i, j, g: g < thr[src[q]],
         'both good (D&D > 1)': lambda q, i, j, g: dnd_good[q, i] > 1 and dnd_good[q, j] > 1}
use = {'dnd': sets['dnd'], 'groups9': sets['groups9'], 'virt66': sets['virt66'], 'behav26': sets['behav26']}
rows = []
for rule, keep in rules.items():
    sel_pairs = [(q, i, j) for (q, i, j), g in zip(pairs, gap) if keep(q, i, j, g)]
    for name, Xs in use.items():
        num = np.zeros((len(qs),) + L.shape[1:3] + (Xs.shape[-1],)); cnt = np.zeros(len(qs))
        for q, i, j in sel_pairs:
            num[q] += (L[q, :, :, i] - L[q, :, :, j])[..., None] * (Xs[q, i] - Xs[q, j])
            cnt[q] += 1
        S = np.where(cnt[:, None, None, None] > 0, num / np.maximum(cnt, 1)[:, None, None, None], np.nan)
        S[np.broadcast_to((np.abs(num).sum((1, 2)) == 0)[:, None, None, :], S.shape)] = np.nan  # no label difference in the kept pairs
        for source in ('machiavelli', 'daily_dilemmas'):
            sep = separation(S, np.ones(len(qs)), (src == source) & ~explore)
            rows.append(dict(source=source, pairs=rule, dims=name, questions=int(((cnt > 0) & (src == source)).sum()), **{k: sep[k] for k in ('rel_q', 'id_q', 'effect')}))
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(200)
print('# Matched-goodness comparison (confirm questions; sharpness removed)\n')
print(pl.DataFrame(rows).sort('source', 'dims', 'pairs'))
