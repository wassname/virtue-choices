"""Per-model profile of what replies appeal to (judge_reason_types.py output), with the same unit-free separation metrics as the label
sweep (split-half reliability across questions, model identification), so reasons and option labels can be compared. -- PI/OpenAI, 2026-10-29"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from normalisation_sweep import load, separation

ROOT = Path(__file__).resolve().parents[2]
J = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/20261029_reason_types_v12/reason_types.jsonl'

qs, dims, models, P, mask, X, src, explore = load()
qi = {q['source_id']: i for i, q in enumerate(qs)}
rows = [json.loads(x) for x in J.open()]
kinds = sorted(rows[0]['probabilities'])
S = np.full((len(qs), len(models), 2, len(kinds)), np.nan)
for r in rows:
    if r['source_id'] in qi:
        S[qi[r['source_id']], models.index(r['model']), r['draw']] = [r['probabilities'][k] for k in kinds]
print(f'{np.isfinite(S[..., 0]).sum()} of {S[..., 0].size} replies judged')
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
for source, name in (('machiavelli', 'Games'), ('daily_dilemmas', 'Everyday')):
    sel = src == source
    mean = np.nanmean(S[sel], (0, 2))
    print(f'\n## {name}: mean probability of each kind of appeal\n')
    print(pl.DataFrame({'model': [m.split('/')[-1][:12] for m in models], **{k: mean[:, j] for j, k in enumerate(kinds)}}))
    sep = {half: separation(S, np.ones(len(qs)), sel & h) for half, h in (('explore', explore), ('confirm', ~explore))}
    print(f"\nseparation: rel_q {sep['explore']['rel_q']:.2f} / {sep['confirm']['rel_q']:.2f} (explore / confirm), id_q {sep['explore']['id_q']:.2f} / {sep['confirm']['id_q']:.2f} (chance 0.17)")
