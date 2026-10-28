"""Do models choose differently along reason kinds, or only write differently? Choice profile: cov of each answer's log p
(sharpness removed: z over options) with the option-level reason-kind probabilities (label_option_reasons.py), per model minus the
mean over models. Stated profile: mean reply-level reason-kind probability (judge_reason_types.py), per model minus mean. -- PI/OpenAI"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from normalisation_sweep import centre, load, p_transforms, profiles, separation, stat

ROOT = Path(__file__).resolve().parents[2]
qs, dims, models, P, mask, X, src, explore = load()
qi = {q['source_id']: i for i, q in enumerate(qs)}
opt = [json.loads(x) for x in (ROOT / 'outputs/20261029_option_reasons_v12/option_reasons.jsonl').open()]
kinds = sorted(opt[0]['probabilities'])
R = np.zeros((len(qs), P.shape[-1], len(kinds)))
for o in opt:
    if o['source_id'] in qi:
        R[qi[o['source_id']], list(qs[qi[o['source_id']]]['options']).index(o['choice_id'])] = [o['probabilities'][k] for k in kinds]
Rc = centre(R, mask[..., None], axis_k=1)
stated = json.loads((ROOT / 'outputs/20261029_reason_types_v12/reason_type_profiles.json').read_text())
short = [m.split('/')[-1][:8] for m in models]
pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
for pt in ('logp_z', 'logp'):
    S = stat(p_transforms(P, mask)[pt], Rc, mask, 'cov')
    for source, name in (('machiavelli', 'Games'), ('daily_dilemmas', 'Everyday')):
        sel = src == source
        chosen = profiles(np.nanmean(S, 2), sel[None].astype(float))[0]  # (M,K) centred across models
        sep = {h: separation(S, np.ones(len(qs)), sel & m) for h, m in (('explore', explore), ('confirm', ~explore))}
        said = np.array([[stated[source]['mean'][m][k] for k in kinds] for m in models])
        said -= said.mean(0)
        print(f'\n## {name}, choices with {pt}: choice-based reason profile (model minus mean; positive = chooses options favoured by this kind more than other models do)')
        print(f"separation of choices: rel_q {sep['explore']['rel_q']:.2f}/{sep['confirm']['rel_q']:.2f}, id_q {sep['explore']['id_q']:.2f}/{sep['confirm']['id_q']:.2f}, effect {sep['confirm']['effect']:.3f}\n")
        keep = [j for j, k in enumerate(kinds) if k not in ('unclear', 'persona')]
        print(pl.DataFrame({'model': short, **{kinds[j]: chosen[:, j] for j in keep}}))
        print('\nsays vs chooses, corr across the 6 models per kind: ' + ', '.join(f'{kinds[j]} {np.corrcoef(said[:, j], chosen[:, j])[0, 1]:+.2f}' for j in keep))
