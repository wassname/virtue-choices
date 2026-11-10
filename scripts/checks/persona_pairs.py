"""Opposed persona pairs: which pairs separate the six default models, and where the models sit in the square of the four D&D personas.

Fourteen persona prompts on DeepSeek (PERSONAS_V12 in scripts/evaluation/run_tiny_eval.py) give eight persona keys (see persona_keying.py).
Writes run.md, persona_pairs.png and alignment_square.png to outputs/20261110_persona_pairs.
Run from scripts/checks: uv run --locked --no-sync python persona_pairs.py -- Claudypoo[fable], 2026-11-10"""

from itertools import combinations
from pathlib import Path

import numpy as np
from tabulate import tabulate

import persona_keying
from normalisation_sweep import load, p_transforms
from persona_keying import BAND, COLOUR, DISPLAY, INK, MARK, MUTED, N_BOOT, ROOT, SOURCES, SURFACE, load_personas, persona_keys, plt, rng, score_source, slope

OUT = ROOT / 'outputs/20261110_persona_pairs'
CORNERS = ['lawful-good', 'chaotic-good', 'lawful-evil', 'chaotic-evil']
PERSONAS = [*persona_keying.PERSONAS, 'rationalist', 'taoist', *CORNERS, 'honest', 'kind', 'cautious', 'bold']
KEYS = {  # name: (personas on the plus side, personas on the minus side)
    'outcome over rule': (['utilitarian'], ['rule-follower']),
    'self over loyalty': (['selfish'], ['loyal']),
    'rationalist over taoist': (['rationalist'], ['taoist']),
    'honest over kind': (['honest'], ['kind']),
    'bold over cautious': (['bold'], ['cautious']),
    'chaotic over lawful': (['chaotic-good', 'chaotic-evil'], ['lawful-good', 'lawful-evil']),
    'good over evil': (['lawful-good', 'chaotic-good'], ['lawful-evil', 'chaotic-evil']),
    'chaotic good over lawful evil': (['chaotic-good'], ['lawful-evil']),
}
DIAGONAL = 'chaotic good over lawful evil'  # the sum of the two keys before it, so it is left out of the joint fit
SQUARE = ('chaotic over lawful', 'good over evil')
N_NULL = 20


def joint(dev, A, idx, cl, explore):
    """dev: (Q, M, 2, K) preference minus the model mean; A: (Q, K, n keys). All keys fitted together.
    -> slopes (M, n keys) with 95% interval, and the share of the repeatable difference that slopes from the explore half predict on the confirm half."""
    G, h = np.einsum('qka,qkb->qab', A, A), np.einsum('qka,qmk->qma', A, dev.mean(2))
    solve = lambda i: np.linalg.solve(G[i].sum(0), h[i].sum(0).T).T
    cls = np.unique(cl[idx])
    boot = [solve(np.concatenate([idx[cl[idx] == c] for c in rng.choice(cls, len(cls))])) for _ in range(N_BOOT)]
    c = idx[~explore[idx]]
    pred = np.einsum('qka,ma->qmk', A[c], solve(idx[explore[idx]]))
    d0, d1 = dev[c][:, :, 0], dev[c][:, :, 1]
    share = ((pred * (d0 + d1)).sum((0, 2)) - (pred ** 2).sum((0, 2))) / (d0 * d1).sum((0, 2))  # 1 - residual/total, both taken across the two answers
    return solve(idx), *np.percentile(boot, [2.5, 97.5], 0), share


def shuffled(idx, n_opts):
    """Each question gets the keys of another question with the same number of options."""
    perm = idx.copy()
    for k in np.unique(n_opts[idx]):
        w = np.flatnonzero(n_opts[idx] == k)
        perm[w] = idx[rng.permutation(w)]
    return perm


def plot_square(sq, names):
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(13, 13.5), facecolor=SURFACE)
    for col, sname in enumerate(SOURCES.values()):
        x, y, p = sq[sname]['x'], sq[sname]['y'], sq[sname]['persona']
        cx, cy = x['est'].mean(), y['est'].mean()
        half = 1.25 * max(np.abs(np.r_[x['lo'], x['hi']] - cx).max(), np.abs(np.r_[y['lo'], y['hi']] - cy).max())
        whole, zoom = axes[0, col], axes[1, col]
        whole.add_patch(plt.Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec=MUTED, ls=(0, (4, 3)), lw=.9))
        for j, pn in enumerate(PERSONAS):
            corner = pn in CORNERS
            whole.plot(*p[j], ls='', marker='D', ms=9 if corner else 5, mew=1.4, mfc=INK if corner else 'none', color=INK if corner else MUTED)
            if corner:
                whole.annotate(pn.replace('-', ' '), p[j], xytext=(0, 9 if p[j, 1] > 0 else -9), textcoords='offset points', ha='center', va='bottom' if p[j, 1] > 0 else 'top',
                               fontsize=10, color=INK, fontweight='bold', zorder=5, bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.75))
            if np.abs(p[j] - (cx, cy)).max() < half:  # inside the dashed square: named in the enlarged panel
                up = p[j, 1] < cy - .8 * half
                zoom.plot(*p[j], ls='', marker='D', ms=7 if corner else 5, mew=1.2, mfc=INK if corner else 'none', color=INK if corner else MUTED)
                zoom.annotate(pn.replace('-', ' '), p[j], xytext=(0, 7 if up else -7), textcoords='offset points', ha='center', va='bottom' if up else 'top',
                              fontsize=8.5, color=MUTED, zorder=5, bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.75))
            elif not corner:
                far = p[j, 0] < cx - half
                whole.annotate(pn.replace('-', ' '), p[j], xytext=(-7, 0) if far else (0, -9), textcoords='offset points', ha='right' if far else 'center',
                               va='center' if far else 'top', fontsize=8.5, color=MUTED)
        whole.plot(x['est'], y['est'], ls='', **{**MARK[sname], 'ms': 5, 'mew': 1.2})
        whole.set_xlim(-.85, .85); whole.set_ylim(-.85, .85)
        whole.set_title(f'{sname}: personas (diamonds) and the six default models', fontsize=11, loc='left')
        for i in range(len(names)):
            zoom.plot([x['lo'][i], x['hi'][i]], [y['est'][i]] * 2, color=COLOUR[sname], lw=1.2)
            zoom.plot([x['est'][i]] * 2, [y['lo'][i], y['hi'][i]], color=COLOUR[sname], lw=1.2)
            zoom.plot(x['est'][i], y['est'][i], **MARK[sname])
        left = x['est'] < np.median(x['est'])
        for side, sel in ((-1, left), (1, ~left)):  # names in a column at each edge, in the order of y
            order = np.flatnonzero(sel)[np.argsort(y['est'][sel])]
            slots = cy + (np.arange(len(order)) - (len(order) - 1) / 2) * half * .3
            under = p[side * (p[:, 0] - cx) > .6 * half, 1]  # personas under this column of names: move the column off them
            slots += max((0, -.15, .15), key=lambda s: np.abs(slots[:, None] + s * half - under).min(initial=9)) * half
            for i, slot in zip(order, slots):
                zoom.annotate(names[i], (x['est'][i], y['est'][i]), xytext=(cx + side * half * .96, slot), ha='left' if side < 0 else 'right', va='center', fontsize=10,
                              bbox=dict(fc=SURFACE, ec='none', pad=1), arrowprops=dict(arrowstyle='-', color=MUTED, lw=.5, shrinkB=5))
        zoom.set_xlim(cx - half, cx + half); zoom.set_ylim(cy - half, cy + half)
        zoom.set_title(f'{sname}, enlarged (the dashed square above); crosses: 95% intervals', fontsize=11, loc='left')
        for ax in (whole, zoom):
            ax.axhline(0, color=MUTED, lw=.8); ax.axvline(0, color=MUTED, lw=.8)
            ax.set_facecolor(SURFACE)
            ax.set_aspect('equal')
            ax.set_xlabel('← lawful        chaotic →'); ax.set_ylabel('← evil        good →')
            ax.tick_params(length=0)
            ax.grid(color=BAND, lw=.6); ax.set_axisbelow(True)
            for edge in ax.spines.values():
                edge.set_visible(False)
    fig.suptitle('Where the six default models sit among DeepSeek personas, on the two D&D axes', x=.01, ha='left', fontsize=12)
    fig.text(.01, .008, 'Axes come from the four black corner personas: x = how far the stated preference of a subject for each option follows (chaotic good + chaotic evil) minus '
             '(lawful good + lawful evil);\ny = the same for (lawful good + chaotic good) minus (lawful evil + chaotic evil). 0 = the mean of the four corners. '
             'Corners are drawn after their own answer noise is removed (it pushes a corner outward).\nGrey diamonds: the other ten personas (not used for the axes); those inside the dashed square are named '
             'in the enlarged panel.\nAll personas are DeepSeek with a persona prompt, one answer per question; default models gave two. Crosses: 95% intervals (games or topics resampled).',
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .97))
    fig.savefig(OUT / 'alignment_square.png', dpi=160, facecolor=SURFACE)


def main():
    qs, dims, models, P, mask, labels, src, explore = load()
    names = [next(v for k, v in DISPLAY.items() if k in m) for m in models]
    ds = names.index('DeepSeek')
    Pp = load_personas(qs, mask, PERSONAS)
    answered = np.isfinite(Pp[:, :, 0, 0])
    ok = answered.all(1)
    T, Tp = p_transforms(P, mask)['logp_z'], p_transforms(np.nan_to_num(Pp), mask)['logp_z'][:, :, 0]
    dev = T - T.mean(1, keepdims=True)
    cl = np.array([q['eval_cluster'] for q in qs])
    n_opts = mask.sum(1)
    keys = persona_keys(Tp, KEYS, PERSONAS)
    by_source = {sname: np.flatnonzero((src == s) & ok) for s, sname in SOURCES.items()}
    res = {k: {sname: score_source(dev, key, idx, cl, explore, n_opts) for sname, idx in by_source.items()} for k, key in keys.items()}
    order = list(np.argsort(res['outcome over rule']['games']['est']))
    OUT.mkdir(parents=True, exist_ok=True)
    persona_keying.plot(res, names, order, KEYS, PERSONAS, OUT / 'persona_pairs.png', nrows=2)

    cell = lambda r, m: (lambda t: f'**{t}**' if r['lo'][m] * r['hi'][m] > 0 else t)(f'{r["est"][m]:+.3f} [{r["lo"][m]:+.2f}, {r["hi"][m]:+.2f}]')
    out = ['# Opposed persona pairs', '',
           f'Run: `{Path(__file__).name}`; {int(ok.sum())} of {len(qs)} questions that all {len(PERSONAS)} personas answered; six default models, 2 answers each; '
           'personas are DeepSeek with a persona prompt, 1 answer. Preference scale `logp_z` (log of the stated percentage, centred over options, unit spread per answer). '
           '-- Claudypoo[fable]', '',
           'On a question with two options this scale is +1 for the top option and -1 for the other, so all everyday scores count top choices only. '
           + '; '.join(f'{s}: {len(i)} questions in {len(np.unique(cl[i]))} {"games" if s == "games" else "topics"}' for s, i in by_source.items())
           + '. Intervals resample games or topics; with few topics the everyday intervals are probably too narrow.', '',
           '## Did DeepSeek take up each persona?', '',
           'Root mean square difference of preference per option against default DeepSeek (answer 1), and how often the top option differs, on the questions that all personas answered.',
           'SHOULD: each persona is farther from default DeepSeek than the two answers of default DeepSeek are from each other. ELSE the prompt did not change the choices.', '']
    rms = lambda a, b, i: np.sqrt(((a[i] - b[i]) ** 2).sum() / mask[i].sum())
    top = lambda A: np.where(mask, A, -1).argmax(-1)
    differs = lambda a, b, i: (top(a) != top(b))[i].mean() * 100
    rows = [{'subject': 'default DeepSeek, answer 2 (noise only)', 'answered': len(qs),
             **{f'{s}: distance': rms(T[:, ds, 0], T[:, ds, 1], i) for s, i in by_source.items()},
             **{f'{s}: top option differs %': differs(P[:, ds, 0], P[:, ds, 1], i) for s, i in by_source.items()}},
            {'subject': 'another default model (mean of 5)', 'answered': len(qs),
             **{f'{s}: distance': np.mean([rms(T[:, ds, 0], T[:, m, 1], i) for m in range(len(names)) if m != ds]) for s, i in by_source.items()},
             **{f'{s}: top option differs %': np.mean([differs(P[:, ds, 0], P[:, m, 1], i) for m in range(len(names)) if m != ds]) for s, i in by_source.items()}}]
    rows += [{'subject': f'persona: {pn}', 'answered': int(answered[:, j].sum()),
              **{f'{s}: distance': rms(T[:, ds, 0], Tp[:, j], i) for s, i in by_source.items()},
              **{f'{s}: top option differs %': differs(P[:, ds, 0], Pp[:, j, 0], i) for s, i in by_source.items()}} for j, pn in enumerate(PERSONAS)]
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt=('', '', '.2f', '.2f', '.0f', '.0f')), '',
            '## Score of each default model on each key', '',
            'Score = slope of (model preference minus six-model mean) on the key. 1.0 = the full gap between the two sides. Bold = the 95% interval '
            '(games or topics resampled) excludes 0. Clear = same sign in games and everyday, and in both the whole interval is outside the control band '
            '(2 sd of the score when the key comes from an unrelated scene). Clear was the rule set before the runs; it is strict (about 4 sd in each source), '
            'so "no clear model" does not show that the models are the same on a key. The summary has two tests that were added after the result was seen.', '']
    summary, shares = [], {}
    for kname, (pos, neg) in KEYS.items():
        g, e = res[kname]['games'], res[kname]['everyday']
        band = 2 * max(g['null_sd'].max(), e['null_sd'].max())
        clear = [m for m in order[::-1] if all(np.sign(r['est'][m]) == np.sign(g['est'][m]) and r['lo'][m] * r['hi'][m] > 0 and min(abs(r['lo'][m]), abs(r['hi'][m])) > band for r in (g, e))]
        rows = [{'model': names[m], 'games [95%]': cell(g, m), 'everyday [95%]': cell(e, m), 'games explore / confirm': f'{g["explore"][m]:+.3f} / {g["confirm"][m]:+.3f}',
                 'everyday explore / confirm': f'{e["explore"][m]:+.3f} / {e["confirm"][m]:+.3f}',
                 'control mean, games / everyday': f'{g["null_mean"][m]:+.3f} / {e["null_mean"][m]:+.3f}', 'clear': 'yes' if m in clear else ''} for m in order[::-1]]
        out += [f'### {kname}: {" + ".join(pos)} minus {" + ".join(neg)}', '', tabulate(rows, headers='keys', tablefmt='pipe'), '',
                f'Control band +-{band:.3f} (2 sd of the control). SHOULD: the control mean is near 0. ELSE the score is partly a lean on option position (in everyday dilemmas: the act against "not" the act). Order of the six models repeats on other games or topics and the other answer: r = {g["reliability"]:.2f} games, '
                f'{e["reliability"]:.2f} everyday. Between games and everyday: r = {np.corrcoef(g["est"], e["est"])[0, 1]:.2f} (6 models).', '']
        spread = lambda r: f'{r["est"].max() - r["est"].min():.3f}'
        adj = lambda r, m, k: r[k][m] - r['null_mean'][m]
        both = [m for m in order[::-1] if adj(g, m, 'est') * adj(e, m, 'est') > 0 and all(adj(r, m, 'lo') * adj(r, m, 'hi') > 0 for r in (g, e))]
        shares[kname] = signal = [1 - (1 / len(pos) + 1 / len(neg)) * rms(T[:, ds, 0], T[:, ds, 1], i) ** 2 / 2 / (keys[kname][i] ** 2).sum() * mask[i].sum() for i in by_source.values()]
        summary.append({'key': kname, 'clear models': ', '.join(f'{names[m]} {g["est"][m]:+.2f} / {e["est"][m]:+.2f}' for m in clear) or 'none',
                        'weaker rule': ', '.join(f'{names[m]} {g["est"][m]:+.2f} / {e["est"][m]:+.2f}' for m in both) or 'none',
                        'p of the spread (games / everyday)': f'{g["range_p"]:.3f} / {e["range_p"]:.3f}',
                        'everyday questions where the key is not 0': int((np.abs(keys[kname][by_source['everyday']]).sum(1) > 0).sum()),
                        'share of the key that is not answer noise (games / everyday)': f'{signal[0]:.2f} / {signal[1]:.2f}',
                        'range of the six scores, games': spread(g), 'everyday': spread(e), 'control band': f'+-{band:.3f}',
                        'model order repeats, r (games / everyday)': f'{g["reliability"]:.2f} / {e["reliability"]:.2f}'})
    out += ['## Summary: which pairs separate the default models', '',
            'Models are shown with their games / everyday score. Two tests were added after the result was seen.',
            '- Weaker rule: after the control mean is subtracted, the score has the same sign in games and everyday, and both 95% intervals exclude 0. '
            f'With no real lean, about {.05 * .05 * .5 * len(KEYS) * len(names):.2f} of the {len(KEYS) * len(names)} model-key cells pass by chance (0.05 x 0.05 x 0.5 each).',
            '- p of the spread: how often the six scores from an unrelated-scene key are spread as widely (largest minus smallest) as the real six scores; '
            f'{persona_keying.N_NULL} shuffles, so {1 / persona_keying.N_NULL:.3f} is the smallest value. The shuffle ignores games and topics, so p is too small by an unknown factor.',
            '',
            'Share of the key that is not answer noise: 1 minus (noise of one answer, from the two answers of default DeepSeek, for each persona in the key) over the '
            'mean square of the key. It assumes that a persona answer is as noisy as a default answer. A score is multiplied by about this share, so a low share pulls it toward 0. '
            'Everyday questions where the key is not 0: the two sides choose differently there; an everyday score rests on these questions only.', '',
            tabulate(summary, headers='keys', tablefmt='pipe'), '',
            '## Do the keys measure different things?', '',
            'Correlation between two keys over all options of all questions.',
            'SHOULD: keys that measure different things have a correlation near 0. ELSE a lean on one key is partly the same lean seen again on the other.', '']
    fit = [k for k in KEYS if k != DIAGONAL]
    A = np.stack([keys[k] for k in fit], -1)
    for sname, idx in by_source.items():
        C = np.corrcoef(np.stack([keys[k][idx].ravel() for k in KEYS]))
        out += [f'{sname}:', '', tabulate([{'key': k, **{k2: C[a, b] for b, k2 in enumerate(KEYS)}} for a, k in enumerate(KEYS)], headers='keys', tablefmt='pipe', floatfmt='+.2f'), '']
    out += ['## All keys fitted together', '',
            f'Slope of (model preference minus six-model mean) on each key, with the other keys in the same fit. `{DIAGONAL}` is left out: it is the sum of the two D&D keys.',
            'SHOULD: a lean that is its own direction keeps an interval that excludes 0 here. ELSE it was another key seen again.', '']
    J = {sname: joint(dev, A, idx, cl, explore) for sname, idx in by_source.items()}
    for sname, (b, lo, hi, _) in J.items():
        rows = [{'model': names[m], **{k: (lambda t: f'**{t}**' if lo[m, a] * hi[m, a] > 0 else t)(f'{b[m, a]:+.3f} [{lo[m, a]:+.2f}, {hi[m, a]:+.2f}]') for a, k in enumerate(fit)}}
                for m in order[::-1]]
        out += [f'{sname}:', '', tabulate(rows, headers='keys', tablefmt='pipe'), '']
    out += ['## How much of the difference between the default models do the keys describe?', '',
            'Share of the repeatable difference of each model from the six-model mean (the part that is the same in its two answers). '
            'Slopes come from the explore half and are scored on the confirm half.',
            'SHOULD: if persona keys describe how the default models differ, the share is large. ELSE most of the repeatable difference is in directions that these personas do not span.',
            'Keys from unrelated scenes give a share below 0: slopes fitted to noise predict worse than no prediction. The real rows pay the same cost, '
            'so read a real row against the unrelated-scene row. One split only: the share of one model is noisy, read the mean.', '']
    sets = {'outcome over rule alone': fit[:1], ' + '.join(fit[:2]): fit[:2], f'all {len(fit)} keys': fit}
    rows = []
    for sname, idx in by_source.items():
        for label, ks in sets.items():
            share = joint(dev, A[..., [fit.index(k) for k in ks]], idx, cl, explore)[3]
            rows.append({'source': sname, 'keys': label, **{names[m]: share[m] for m in order[::-1]}, 'mean': share.mean()})
        null = []
        for _ in range(N_NULL):
            perm = idx.copy()
            perm[:] = shuffled(idx, n_opts)
            A0 = A.copy()
            A0[idx] = A[perm]
            null.append(joint(dev, A0, idx, cl, explore)[3])
        rows.append({'source': sname, 'keys': f'all {len(fit)} keys from unrelated scenes (mean of {N_NULL})', **{names[m]: np.mean(null, 0)[m] for m in order[::-1]}, 'mean': np.mean(null)})
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.3f'), '',
            '## The square of the four D&D personas', '',
            f'x = {SQUARE[0]}, y = {SQUARE[1]}. 0 = the mean of the four corner personas (not the six-model mean). Each axis is fitted alone.',
            'SHOULD: the two corners of a side are on the same side of 0, and personas that were not used for the axes land where their names say '
            '(selfish toward evil, rule-follower toward lawful). ELSE the axis does not measure what its name says.', '']
    origin = Tp[:, [PERSONAS.index(c) for c in CORNERS]].mean(1)
    sq = {sname: {'x': score_source(T - origin[:, None, None], keys[SQUARE[0]], idx, cl, explore, n_opts),
                  'y': score_source(T - origin[:, None, None], keys[SQUARE[1]], idx, cl, explore, n_opts),
                  'persona': np.stack([slope(Tp[idx] - origin[idx, None], keys[a][idx]) for a in SQUARE], 1)} for sname, idx in by_source.items()}
    push = {s: [.5 * (1 - shares[a][n]) for a in SQUARE] for n, s in enumerate(by_source)}  # a corner's own answer noise is also in the key
    for s in sq:
        for b, a in enumerate(SQUARE):
            sq[s]['persona'][[PERSONAS.index(c) for c in CORNERS], b] -= [push[s][b] * ((c in KEYS[a][0]) - (c in KEYS[a][1])) for c in CORNERS]
    plot_square(sq, names)
    plain = lambda r, m: f'{r["est"][m]:+.3f} [{r["lo"][m]:+.2f}, {r["hi"][m]:+.2f}]'
    rows = [{'subject': f'persona: {pn}', **{f'{s}: {a}': f'{sq[s]["persona"][j, b]:+.3f}' for s in sq for b, a in enumerate(('x', 'y'))}} for j, pn in enumerate(PERSONAS)]
    rows += [{'subject': names[m], **{f'{s}: {a}': plain(sq[s][a], m) for s in sq for a in ('x', 'y')}} for m in order[::-1]]
    out += [tabulate(rows, headers='keys', tablefmt='pipe'), '',
            'A corner persona carries its own answer noise into the axis that it helps to define, which pushes it outward by half of the noise share of the axis. '
            'This push is removed from the corner rows above and from the plot: '
            + '; '.join(f'{s} x {push[s][0]:.2f}, y {push[s][1]:.2f}' for s in push) + '. It assumes that a persona answer is as noisy as a default answer. '
            'All points are also pulled toward 0 by the same noise (multiply by the share in the summary table); this is not corrected.', '',
            '## Each D&D axis from one pair of corners', '',
            'Score as in the second section (0 = the six-model mean), with the key from two of the four corner personas.',
            'SHOULD: a lean on an axis has the same sign from both pairs. ELSE the lean belongs to one persona, not to the axis.', '']
    pairs = {'chaotic over lawful, good pair': (['chaotic-good'], ['lawful-good']), 'chaotic over lawful, evil pair': (['chaotic-evil'], ['lawful-evil']),
             'good over evil, lawful pair': (['lawful-good'], ['lawful-evil']), 'good over evil, chaotic pair': (['chaotic-good'], ['chaotic-evil'])}
    rows = [(lambda r: {'key': k, 'source': s, **{names[m]: cell(r, m) for m in order[::-1]}})(score_source(dev, key, idx, cl, explore, n_opts))
            for k, key in persona_keys(Tp, pairs, PERSONAS).items() for s, idx in by_source.items()]
    out += [tabulate(rows, headers='keys', tablefmt='pipe'), '',
            f'Plots: {OUT / "persona_pairs.png"}', f'{OUT / "alignment_square.png"}', f'This file: {OUT / "run.md"}']
    (OUT / 'run.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
