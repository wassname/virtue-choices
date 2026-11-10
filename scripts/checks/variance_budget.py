"""Where the spread of stated preference comes from, and how much of the model-specific part is a profile.

Table 1 (no labels): the spread between options within a question splits into a part all models share, a model-specific part that
repeats on the model's other answer, and answer noise. Table 2: a profile (spider chart, D&D point) is the part of a model's
difference that is the same in all scenes; its share of the model-specific repeatable variance, per label set.
Run from scripts/checks: uv run --locked --no-sync python variance_budget.py -- Claudypoo[fable], 2026-11-10"""

from pathlib import Path

import numpy as np
from tabulate import tabulate

from normalisation_sweep import dim_sets, load, p_transforms, stat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/20261110_diagnosis/variance_budget.md'
SOURCES = {'machiavelli': 'games', 'daily_dilemmas': 'everyday'}
N_BOOT, N_SPLITS = 500, 200
rng = np.random.default_rng(20261110)


def split(a, b, b_same=None):
    """a, b: (q, M, K) the two answers, centred over options -> shares [same for all models, model-specific repeatable, answer noise].
    Every product is between different answers, so answer noise and option-order effects do not enter the first two shares."""
    M = a.shape[1]
    total = ((a ** 2).sum() + (b ** 2).sum()) / 2
    same_model = (a * (b if b_same is None else b_same)).sum()
    other_model = ((a.sum(1) * b.sum(1)).sum() - (a * b).sum()) / (M - 1)
    return np.array([other_model, same_model - other_model, total - same_model]) / total


def resample(idx, cl):
    cls = np.unique(cl[idx])
    return np.concatenate([idx[cl[idx] == c] for c in rng.choice(cls, len(cls))])


def profile(S, idx, cl):
    """S: (Q, M, 2, D) per-scene covariance of preference with a label -> (share of model-specific repeatable variance that is
    constant across scenes, split-half reliability of the model means). Halves are whole games or topics and use different answers."""
    dev = S[idx] - np.nanmean(S[idx], 1, keepdims=True)
    repeatable = np.nansum(np.nanmean(dev[:, :, 0] * dev[:, :, 1], 0))
    cls = np.unique(cl[idx]); const, rel = [], []
    for _ in range(N_SPLITS):
        h = np.isin(cl[idx], rng.permutation(cls)[:len(cls) // 2])
        a0, a1, b0, b1 = (np.nanmean(dev[sel][:, :, r], 0) for sel in (h, ~h) for r in (0, 1))  # (M, D) each
        const.append(np.nansum(a0 * b1 + a1 * b0) / 2)
        r = np.nanmean([np.corrcoef((a0 + a1)[:, d], (b0 + b1)[:, d])[0, 1] for d in range(dev.shape[-1])])
        rel.append(2 * r / (1 + r))
    return np.mean(const) / repeatable, np.mean(rel)


def main():
    qs, dims, models, P, mask, X, src, _ = load()
    sets, _ = dim_sets(dims, X, mask, src)
    cl = np.array([q['eval_cluster'] for q in qs])
    Ts = p_transforms(P, mask)
    out = ['# Where the spread of stated preference comes from', '',
           f'Run: `{Path(__file__).name}` on `outputs/20261027_random1000_v12` ({len(qs)} questions, {len(models)} models, 2 answers each). -- Claudypoo[fable]', '',
           '## 1. Share of the spread between options, within a question (no labels)', '',
           'SHOULD: the three shares sum to 1; the control (answer 2 taken from another model) reads 0.00 for the model-specific share. '
           'ELSE the estimator counts something that is not tied to the model.', '']
    rows = []
    for s, sname in SOURCES.items():
        idx = np.flatnonzero(src == s)
        for t in ('logp_z', 'p', 'rank'):
            a, b = Ts[t][idx][:, :, 0], Ts[t][idx][:, :, 1]
            est = split(a, b)
            boot = np.array([split(Ts[t][i][:, :, 0], Ts[t][i][:, :, 1]) for i in (resample(idx, cl) for _ in range(N_BOOT))])
            lo, hi = np.percentile(boot, [2.5, 97.5], 0)
            shift = rng.integers(1, len(models), len(idx))
            null = split(a, b, b_same=np.stack([np.roll(b[q], shift[q], 0) for q in range(len(idx))]))[1]
            rows.append({'source': sname, 'preference scale': t, 'model-specific, repeatable': f'{est[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]',
                         'same for all models': f'{est[0]:.3f}', 'answer noise': f'{est[2]:.3f}',
                         'signal share of one answer after the shared part is removed': f'{est[1] / (est[1] + est[2]):.2f}', 'control': f'{null:+.3f}'})
    out += [tabulate(rows, headers='keys', tablefmt='pipe'), '',
            'Scales: `logp_z` = log of the stated percentage, centred over options, unit spread per answer; `p` = percentage; `rank` = wins minus losses. '
            'Interval: games or topics resampled.', '']
    top = np.where(mask[:, None, None, :], P, -1).argmax(-1)  # (Q, M, 2)
    rows = []
    for s, sname in SOURCES.items():
        t = top[src == s]
        other = np.mean([(t[:, m, 0] == t[:, n, 1]).mean() for m in range(len(models)) for n in range(len(models)) if m != n])
        rows.append({'source': sname, 'same model, other answer': (t[:, :, 0] == t[:, :, 1]).mean(), 'another model, other answer': other,
                     'all 12 answers': (t.reshape(len(t), -1) == t[:, :1, 0]).all(1).mean()})
    out += ['Share of questions with the same top option:', '', tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.3f'), '',
            '## 2. How much of the model-specific part is a profile', '',
            'SHOULD: if models had stable virtue profiles the share would be large (near 1). A share near 0 with high reliability means '
            'the profile is measured well and is small; a share near 0 with low reliability would mean we cannot tell.', '']
    rows = []
    for name, Xl in sets.items():
        S = stat(Ts['logp_z'], Xl, mask, 'cov')
        row = {'label set': name}
        for s, sname in SOURCES.items():
            share, rel = profile(S, np.flatnonzero(src == s), cl)
            row[f'{sname}: share constant across scenes'] = share
            row[f'{sname}: reliability of model means'] = rel
        rows.append(row)
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.3f'), '',
            'Label sets as in `normalisation_sweep.dim_sets`: goodness = first component of the 66 direct virtues; style9 = 9 virtue groups with goodness removed.', '',
            '## 3. Questions with one clearly best option against questions where two good options compete', '',
            'Questions in thirds by the goodness label: gap between the best and the second-best option.',
            'SHOULD: if "one good option among bad ones" is the whole problem, the small-gap third has a large profile share. ELSE that third has more '
            'model-specific spread but it still changes from scene to scene.', '']
    good = np.where(mask, sets['goodness'][..., 0], -np.inf)
    gap = np.sort(good, 1)[:, -1] - np.sort(good, 1)[:, -2]
    S = {k: stat(Ts['logp_z'], sets[k], mask, 'cov') for k in ('virt66', 'style9')}
    rows = []
    for s, sname in SOURCES.items():
        idx = np.flatnonzero(src == s)
        third = np.digitize(gap[idx], np.quantile(gap[idx], [1 / 3, 2 / 3]))
        for b, bname in enumerate(('small gap: two good options compete', 'middle', 'large gap: one clearly best option')):
            i = idx[third == b]
            sh = split(Ts['logp_z'][i][:, :, 0], Ts['logp_z'][i][:, :, 1])
            rows.append({'source': sname, 'third': bname, 'questions': len(i), 'all 12 answers same top option': (top[i].reshape(len(i), -1) == top[i][:, :1, 0]).all(1).mean(),
                         'answers that pick the label-best option': (top[i] == good[i].argmax(1)[:, None, None]).mean(),
                         'same for all models': sh[0], 'model-specific, repeatable': sh[1], 'answer noise': sh[2],
                         'profile share, 66 virtues': profile(S['virt66'], i, cl)[0], 'profile share, 9 groups without goodness': profile(S['style9'], i, cl)[0]})
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.2f'), '', f'This file: {OUT}']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(out) + '\n')
    print('\n'.join(out), f'\n\nwrote {OUT}')


if __name__ == '__main__':
    main()
