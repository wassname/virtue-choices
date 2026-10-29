"""Neighbour-projection dev test (v14 'another instance of you') against v12 on the same 200 questions and 6 models.
(1) label-free diversity: single-answer JS between models minus within a model, with question bootstrap; top-choice disagreement.
(2) per model, D&D lawful/good of its choices (cov of sharpness-removed log p with centred labels), v14 minus v12: does projection
make answers less socially desirable, as the survey trick predicts? -- PI/OpenAI, 2026-10-29"""

import itertools
import json

import numpy as np
import polars as pl

from normalisation_sweep import dim_sets, load, p_transforms, stat
from prompt_diversity import complete, js, load as load_runs, top

qs, dims, models, P, mask, X, src, explore = load()
sets, _ = dim_sets(dims, X, mask, src)
v12 = load_runs(['outputs/20261027_random1000_v12', 'outputs/20261027_random1000_v12_haiku'])
v14 = load_runs(['outputs/20261029_neighbour_v14'])
six = sorted({m for _, m in v14})
common = sorted(complete(v12, six) & complete(v14, six))
rng = np.random.default_rng(1)
rows = []
for source in ('machiavelli', 'daily_dilemmas'):
    qsel = [q for q in common if q.startswith(source)]
    for name, data in (('v12', v12), ('v14 neighbour', v14)):
        for mname, D in (('JS', js), ('top differs', top)):
            b = np.array([np.mean([D(data[q, a][i], data[q, c][j]) for a, c in itertools.combinations(six, 2) for i in (0, 1) for j in (0, 1)]) for q in qsel])
            w = np.array([np.mean([D(data[q, m][0], data[q, m][1]) for m in six]) for q in qsel])
            boot = [(b - w)[rng.integers(0, len(b), len(b))].mean() for _ in range(2000)]
            rows.append(dict(source=source, prompt=name, metric=mname, questions=len(qsel), between=b.mean(), within=w.mean(), excess=(b - w).mean(),
                             lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(3); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(14)
print('# Neighbour prompt (v14) vs v12, same questions and models\n')
print(pl.DataFrame(rows))

# (2) D&D shift per model
qi = {q['source_id']: i for i, q in enumerate(qs)}
P14 = np.full_like(P, np.nan)
for (q, m), dr in v14.items():
    if q in qi and m in models and len(dr) == 2:
        a = qi[q]
        for r in (0, 1):
            P14[a, models.index(m), r, :int(mask[a].sum())] = [dr[r][o] / 100 for o in qs[a]['options']]
sel_q = np.zeros(len(qs), bool); sel_q[[qi[q] for q in common]] = True
out = []
for source in ('machiavelli', 'daily_dilemmas'):
    sel = sel_q & (src == source)
    ok14 = np.isfinite(P14[..., 0]).all((1, 2))
    sel = sel & ok14
    for name, PP in (('v12', P), ('v14', np.nan_to_num(P14))):
        S = np.nanmean(np.nanmean(stat(p_transforms(PP, mask)['logp_z'], sets['dnd'], mask, 'cov'), 2)[sel], 0)  # (M,2)
        for j, m in enumerate(models):
            out.append(dict(source=source, model=m.split('/')[-1][:10], prompt=name, lawful=S[j, 0], good=S[j, 1]))
df = pl.DataFrame(out).pivot(on='prompt', index=['source', 'model'], values=['lawful', 'good'])
df = df.with_columns(d_lawful=pl.col('lawful_v14') - pl.col('lawful_v12'), d_good=pl.col('good_v14') - pl.col('good_v12'))
print('\n## D&D of chosen options (sharpness removed; question-centred, so 0 = no preference), v14 minus v12\n')
print(df.select('source', 'model', 'lawful_v12', 'lawful_v14', 'd_lawful', 'good_v12', 'good_v14', 'd_good'))
print('\nspread across models (sd):')
print(df.group_by('source').agg(pl.col('good_v12').std(), pl.col('good_v14').std(), pl.col('lawful_v12').std(), pl.col('lawful_v14').std()))
