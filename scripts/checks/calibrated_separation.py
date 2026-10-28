"""Same split-half reliability / size test, on the calibrated per-answer log p used by the published plots (calibration_heldout
quantile maps; DeepSeek identity), against raw log p and per-answer z-scored log p. -- PI/OpenAI, 2026-10-29"""

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
C = np.zeros_like(P)
for a, q in enumerate(qs):
    k = len(q['options'])
    for b, m in enumerate(models):
        for r in (0, 1):
            C[a, b, r, :k] = calibrated_logp(q, m, maps, fold, (r,))
m4 = mask[:, None, None, :]
Ts = p_transforms(P, mask)
Ts['calibrated'] = centre(C, m4)
rows = []
for name in ('goodness', 'dnd', 'groups9', 'virt66', 'style9', 'behav26'):
    for pt in ('logp', 'logp_z', 'calibrated'):
        S = stat(Ts[pt], sets[name], mask, 'cov')
        for source in ('machiavelli', 'daily_dilemmas'):
            sep = separation(S, np.ones(len(qs)), (src == source) & ~explore)
            rows.append(dict(source=source, dims=name, probs=pt, rel_q=sep['rel_q'], effect=sep['effect']))
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(200); pl.Config.set_tbl_cols(12)
print('# Calibrated vs raw vs z-scored log p (confirm questions)\n')
print(pl.DataFrame(rows).pivot(on='probs', index=['source', 'dims'], values=['rel_q', 'effect']).sort('source', 'dims'))
