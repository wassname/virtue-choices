"""Split-half reliability of a Kendall-tau spider built from comparative labels (all pairs, and pairs goodness does not decide). -- PI/OpenAI, 2026-10-29"""
import sys; sys.argv=['x']
src_code = open('comparative_fit.py').read().split("dev = ((C - C.mean")[0]
exec(src_code.replace("print(", "(lambda *a, **k: None)("))
# per-answer tau S[q,m,r,d]
S = np.full((len(qs), len(models), 2, len(D)), np.nan)
for a in np.flatnonzero(have):
    k = int(mask[a].sum()); i, j = iu[k]
    sx = np.sign(Qd[a, i] - Qd[a, j])
    for r in (0, 1):
        S[a, :, r] = np.sign(P[a][:, r, i] - P[a][:, r, j]) @ sx / len(i)
# goodness-conditional tau: only option pairs the Goodness ranking does not separate much (|dq_good| < 0.2)
g = D.index('Goodness')
Sc = np.full_like(S, np.nan)
for a in np.flatnonzero(have):
    k = int(mask[a].sum()); i, j = iu[k]
    keep = np.abs(Qd[a, i, g] - Qd[a, j, g]) < .2
    if keep.sum() == 0: continue
    i2, j2 = i[keep], j[keep]
    sx = np.sign(Qd[a, i2] - Qd[a, j2])
    for r in (0, 1):
        Sc[a, :, r] = np.sign(P[a][:, r, i2] - P[a][:, r, j2]) @ sx / len(i2)
rows = []
for source in ('machiavelli', 'daily_dilemmas'):
    for name, SS in (('tau, all pairs', S), ('tau, pairs goodness does not decide', Sc)):
        sel = (src == source) & ~explore & np.isfinite(SS[:, 0, 0, 0])
        sep = separation(SS[..., :11], np.ones(len(qs)), sel)  # drop Goodness spoke
        # per-spoke split-half reliability (corr over models, mean over 200 splits)
        rows.append(dict(source=source, statistic=name, questions=int(sel.sum()), rel_q=sep['rel_q'], id_q=sep['id_q'], reliable_spread=sep['reliable_spread']))
print(pl.DataFrame(rows))
# per-spoke reliability for tau all pairs, both sources, all questions (explore+confirm)
from normalisation_sweep import profiles, corr_models
rng = np.random.default_rng(0)
for source in ('machiavelli', 'daily_dilemmas'):
    idx = np.flatnonzero((src == source) & have)
    Sm = np.nanmean(S, 2)
    rs = []
    for _ in range(200):
        p = rng.permutation(idx); WA = np.zeros((1, len(qs))); WB = np.zeros((1, len(qs)))
        WA[0, p[:len(p)//2]] = 1; WB[0, p[len(p)//2:]] = 1
        A, B = profiles(Sm, WA)[0], profiles(Sm, WB)[0]
        rs.append([np.corrcoef(A[:, d], B[:, d])[0, 1] for d in range(len(D))])
    print(source, 'per-spoke split-half reliability (all questions):', ', '.join(f'{d} {v:.2f}' for d, v in zip(D, np.mean(rs, 0))))

# relative tau table: model minus mean of the six, mean of both answers, 95% interval by question bootstrap
print('\n## Relative tau (model minus 6-model mean, x100), all questions; spokes with split-half reliability >= .6\n')
rel = {'machiavelli': ['Kindness', 'Humility', 'Forbearance', 'Self-control', 'Wisdom', 'Self-interest', 'Truthfulness', 'Rules', 'Goodness'],
       'daily_dilemmas': ['Truthfulness', 'Justice', 'Inquiry', 'Wisdom', 'Self-interest', 'Goodness']}
Sm = np.nanmean(S, 2)
for source in ('machiavelli', 'daily_dilemmas'):
    idx = np.flatnonzero((src == source) & have)
    dev = Sm[idx] - np.nanmean(Sm[idx], 1, keepdims=True)
    mean = np.nanmean(dev, 0)
    boot = np.array([np.nanmean(dev[rng.integers(0, len(idx), len(idx))], 0) for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5], 0)
    cols = {d: [f"{100 * mean[m, D.index(d)]:+.1f}{'*' if lo[m, D.index(d)] > 0 or hi[m, D.index(d)] < 0 else ''}" for m in range(len(models))] for d in rel[source]}
    print(f'### {source} ({len(idx)} questions; * = 95% interval excludes 0)\n')
    print(pl.DataFrame({'model': [m.split('/')[-1][:10] for m in models], **cols}))
    print()
