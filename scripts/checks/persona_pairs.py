"""Opposed persona pairs: which pairs separate the six default models, and where the models sit in the square of the four D&D personas.

Fourteen persona prompts on DeepSeek (PERSONAS_V12 in scripts/evaluation/run_tiny_eval.py) give eight persona keys (see persona_keying.py).
Runs: outputs/*_persona-v2_<persona>. The earlier runs (outputs/*_persona_<persona>) had a prompt that also said "with your own values"; they are used for one comparison section.
Writes run.md, persona_pairs.png and alignment_square.png to outputs/20261110_persona-v2_pairs.
Run from scripts/checks: uv run --locked --no-sync python persona_pairs.py -- Claudypoo[fable], 2026-11-10"""

import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
from tabulate import tabulate

import persona_keying
from normalisation_sweep import load, p_transforms
from persona_keying import BAND, COLOUR, DISPLAY, INK, MARK, MUTED, N_BOOT, N_SPLITS, ROOT, SOURCES, SURFACE, load_personas, persona_keys, plt, rng, score_source, slope

OUT = ROOT / 'outputs/20261110_persona-v2_pairs'
TAG, EARLIER = 'persona-v2', 'persona'  # run directories: outputs/*_<tag>_<persona>
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


def joint(dev, A, idx, cl):
    """dev: (Q, M, 2, K) preference minus the model mean; A: (Q, K, n keys). All keys fitted together -> slopes (M, n keys) with 95% interval."""
    G, h = np.einsum('qka,qkb->qab', A, A), np.einsum('qka,qmk->qma', A, dev.mean(2))
    solve = lambda i: np.linalg.solve(G[i].sum(0), h[i].sum(0).T).T
    cls = np.unique(cl[idx])
    boot = [solve(np.concatenate([idx[cl[idx] == c] for c in rng.choice(cls, len(cls))])) for _ in range(N_BOOT)]
    return solve(idx), *np.percentile(boot, [2.5, 97.5], 0)


def held_out_share(dev, A, idx, cl):
    """Slopes fitted on half of the games or topics predict the other half, over N_SPLITS random halves.
    -> (M,) share of the repeatable difference (the part that is the same in the two answers of a model) that the keys predict."""
    G, h = np.einsum('qka,qkb->qab', A, A), np.einsum('qka,qmk->qma', A, dev.mean(2))
    cls = np.unique(cl[idx])
    num = den = 0
    for _ in range(N_SPLITS):
        fit = np.isin(cl[idx], rng.permutation(cls)[:len(cls) // 2])
        c = idx[~fit]
        pred = np.einsum('qka,ma->qmk', A[c], np.linalg.solve(G[idx[fit]].sum(0), h[idx[fit]].sum(0).T).T)
        d0, d1 = dev[c][:, :, 0], dev[c][:, :, 1]
        num += (pred * (d0 + d1)).sum((0, 2)) - (pred ** 2).sum((0, 2))  # total minus residual, both taken across the two answers
        den += (d0 * d1).sum((0, 2))
    return num / den


def names_conflict(tag, persona):
    """Share of replies whose hidden reasoning has the word tension, conflict or contradict. Scenes about a conflict match too (about 0.2)."""
    (run,) = (ROOT / 'outputs').glob(f'*_{tag}_{persona}')
    texts = [json.loads(line)['response']['choices'][0]['message']['reasoning'] or '' for f in run.glob('*/receipts.jsonl') for line in f.open()]
    return np.mean([bool(re.search('tension|conflict|contradict', t, re.I)) for t in texts])


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
        box = half * np.array((.2, .05))  # about half the width and half the height of a persona name in the enlarged panel
        inside = np.abs(p - (cx, cy)).max(1) < half  # inside the dashed square: named in the enlarged panel
        taken = [*p[inside], *np.c_[x['est'], y['est']]]  # points and names that a new name must not cover
        for j, pn in enumerate(PERSONAS):
            corner = pn in CORNERS
            whole.plot(*p[j], ls='', marker='D', ms=9 if corner else 5, mew=1.4, mfc=INK if corner else 'none', color=INK if corner else MUTED)
            if corner:
                whole.annotate(pn.replace('-', ' '), p[j], xytext=(0, 9 if p[j, 1] > 0 else -9), textcoords='offset points', ha='center', va='bottom' if p[j, 1] > 0 else 'top',
                               fontsize=10, color=INK, fontweight='bold', zorder=5, bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.75))
            if inside[j]:
                zoom.plot(*p[j], ls='', marker='D', ms=7 if corner else 5, mew=1.2, mfc=INK if corner else 'none', color=INK if corner else MUTED, clip_on=False)
                centre = lambda side: p[j] + 1.4 * box * side
                free = lambda side: -1 if (np.abs(centre(side) - (cx, cy)) + box > half).any() else min(1.4, min(np.abs((centre(side) - t) / box).max() for t in taken))
                side = np.array(max(((0, -1), (0, 1), (1, 0), (-1, 0)), key=free))  # below, unless the name would leave the panel or cover a point or another name
                taken += [centre(side), centre(side) - (box[0], 0), centre(side) + (box[0], 0)]
                zoom.annotate(pn.replace('-', ' '), p[j], xytext=7 * side, textcoords='offset points', ha=('center', 'left', 'right')[side[0]], va=('center', 'bottom', 'top')[side[1]],
                              fontsize=8.5, color=MUTED, zorder=5, bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.75))
            elif not corner:
                far = p[j, 0] < cx - half
                whole.annotate(pn.replace('-', ' '), p[j], xytext=(-7, 0) if far else (0, -9), textcoords='offset points', ha='right' if far else 'center',
                               va='center' if far else 'top', fontsize=8.5, color=MUTED, zorder=5, bbox=dict(fc=SURFACE, ec='none', pad=.5, alpha=.75))
        whole.plot(x['est'], y['est'], ls='', **{**MARK[sname], 'ms': 5, 'mew': 1.2})
        whole.set_xlim(-.85, .85); whole.set_ylim(-.85, .85)
        whole.set_title(f'{sname}: personas (diamonds) and the six default models', fontsize=11, loc='left')
        for i in range(len(names)):
            zoom.plot([x['lo'][i], x['hi'][i]], [y['est'][i]] * 2, color=COLOUR[sname], lw=1.2)
            zoom.plot([x['est'][i]] * 2, [y['lo'][i], y['hi'][i]], color=COLOUR[sname], lw=1.2)
            zoom.plot(x['est'][i], y['est'][i], zorder=6, **MARK[sname])
        pts = np.c_[x['est'], y['est']]
        ends = np.c_[np.full(len(names), cx + half * 1.06), cy + (np.arange(len(names)) - (len(names) - 1) / 2) * half * .3]  # model names in one column right of the panel

        def line_cost(o):  # total length of the lines (shortest: no two cross), and a large cost for a line that passes close to the marker of another model
            a = pts[list(o)]
            along = np.clip(np.einsum('kjd,kd->kj', pts[None] - a[:, None], ends - a) / ((ends - a) ** 2).sum(1)[:, None], 0, 1)
            gap = np.linalg.norm(a[:, None] + along[..., None] * (ends - a)[:, None] - pts[None], axis=-1)  # (line, marker)
            gap[np.arange(len(o)), list(o)] = np.inf
            return np.linalg.norm(ends - a, axis=1).sum() + 9 * (gap < .05 * half).sum()
        for end, i in zip(ends, min(permutations(range(len(names))), key=line_cost)):
            zoom.annotate(names[i], pts[i], xytext=end, ha='left', va='center', fontsize=10,
                          annotation_clip=False, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.5, shrinkB=5))
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
             '(lawful good + lawful evil);\ny = the same for (lawful good + chaotic good) minus (lawful evil + chaotic evil). 0 = the mean of the four corners; 1.0 = the full difference between the two sides of an axis.\n'
             'A corner helps to define its own axis, so the random part of its answers pushes it outward. That push is removed, so the two sides of an axis are drawn less than 1.0 apart.\n'
             'Grey diamonds: the other ten personas (not used for the axes); those inside the dashed square are named in the enlarged panel.\n'
             'All personas are DeepSeek with a persona prompt, one answer per question; default models gave two. Crosses: 95% intervals (games or topics resampled).',
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .97))
    fig.savefig(OUT / 'alignment_square.png', dpi=160, facecolor=SURFACE)


def main():
    qs, dims, models, P, mask, labels, src, explore = load()
    names = [next(v for k, v in DISPLAY.items() if k in m) for m in models]
    ds = names.index('DeepSeek')
    Pp = load_personas(qs, mask, PERSONAS, TAG)
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
    sides = lambda k: 1 / len(KEYS[k][0]) + 1 / len(KEYS[k][1])
    noise_share = lambda key, n, i: 1 - n * rms(T[:, ds, 0], T[:, ds, 1], i) ** 2 / 2 / (key[i] ** 2).sum() * mask[i].sum()  # n: 1/personas on the plus side + 1/on the minus side
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
        shares[kname] = signal = [noise_share(keys[kname], sides(kname), i) for i in by_source.values()]
        summary.append({'key': kname, 'clear models': ', '.join(f'{names[m]} {g["est"][m]:+.2f} / {e["est"][m]:+.2f}' for m in clear) or 'none',
                        'weaker rule': ', '.join(f'{names[m]} {g["est"][m]:+.2f} / {e["est"][m]:+.2f}' for m in both) or 'none',
                        'p of the spread (games / everyday)': f'{g["range_p"]:.3f} / {e["range_p"]:.3f}',
                        'everyday questions where the key is not 0': int((np.abs(keys[kname][by_source['everyday']]).sum(1) > 0).sum()),
                        'share of the key that is not answer noise (games / everyday)': f'{signal[0]:.2f} / {signal[1]:.2f}',
                        'range of the six scores, games': spread(g), 'everyday': spread(e), 'control band': f'+-{band:.3f}',
                        'model order repeats, r (games / everyday)': f'{g["reliability"]:.2f} / {e["reliability"]:.2f}'})
    far = {s: [rms(T[:, ds, 0], Tp[:, j], i) for j in range(len(PERSONAS))] for s, i in by_source.items()}  # distance of each persona to default DeepSeek
    nearer = [np.mean([far[s][PERSONAS.index(pn)] for pn in neg]) - np.mean([far[s][PERSONAS.index(pn)] for pn in pos]) for pos, neg in KEYS.values() for s in by_source]
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
            'Is a score partly made by the way the key is built? The keys come from DeepSeek. When the plus side is nearer to default DeepSeek than the minus side, '
            'the key is close to "default DeepSeek minus the far persona", so it holds part of what is particular to default DeepSeek.',
            'SHOULD: the score of a model does not follow how much nearer the plus side is (r near 0). ELSE the score of that model on a lopsided key is partly an effect of the method.',
            f'r over the {len(nearer)} key-source cells (keys share personas, so these are not {len(nearer)} independent values): '
            + ', '.join(f'{names[m]} {np.corrcoef(nearer, [res[k][s]["est"][m] for k in KEYS for s in by_source])[0, 1]:+.2f}' for m in order[::-1]) + '.',
            'With DeepSeek left out of the model mean, the score of another model is its score plus one fifth of the DeepSeek score: a change of at most '
            f'{max(abs(res[k][s]["est"][ds]) for k in KEYS for s in by_source) / 5:.3f}.', '',
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
    J = {sname: joint(dev, A, idx, cl) for sname, idx in by_source.items()}
    for sname, (b, lo, hi) in J.items():
        rows = [{'model': names[m], **{k: (lambda t: f'**{t}**' if lo[m, a] * hi[m, a] > 0 else t)(f'{b[m, a]:+.3f} [{lo[m, a]:+.2f}, {hi[m, a]:+.2f}]') for a, k in enumerate(fit)}}
                for m in order[::-1]]
        out += [f'{sname}:', '', tabulate(rows, headers='keys', tablefmt='pipe'), '']
    out += ['## How much of the difference between the default models do the keys describe?', '',
            'Share of the repeatable difference of each model from the six-model mean (the part that is the same in its two answers). '
            f'Slopes come from a random half of the games or topics and are scored on the other half; summed over {N_SPLITS} random halves.',
            'SHOULD: if persona keys describe how the default models differ, the share is large. ELSE most of the repeatable difference is in directions that these personas do not span.',
            'Keys from unrelated scenes give a share below 0: slopes fitted to noise predict worse than no prediction. The real rows pay the same cost, '
            'so read a real row against the unrelated-scene row. With 17 topics the everyday share of one model is noisy: read the mean.', '']
    sets = {'outcome over rule alone': fit[:1], ' + '.join(fit[:2]): fit[:2], f'all {len(fit)} keys': fit}
    rows = []
    for sname, idx in by_source.items():
        for label, ks in sets.items():
            share = held_out_share(dev, A[..., [fit.index(k) for k in ks]], idx, cl)
            rows.append({'source': sname, 'keys': label, **{names[m]: share[m] for m in order[::-1]}, 'mean': share.mean()})
        null = []
        for _ in range(N_NULL):
            perm = idx.copy()
            perm[:] = shuffled(idx, n_opts)
            A0 = A.copy()
            A0[idx] = A[perm]
            null.append(held_out_share(dev, A0, idx, cl))
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
    halves = list(persona_keys(Tp, pairs, PERSONAS).values())
    out += [tabulate(rows, headers='keys', tablefmt='pipe'), '',
            'Correlation of the two pairs of one axis, over all options, and in brackets the value that answer noise alone would leave if both pairs measured one direction. '
            'SHOULD: near the value in brackets. ELSE DeepSeek does not play the axis as one direction. '
            + '; '.join(f'{a}: ' + ', '.join(f'{np.corrcoef(halves[2 * b][i].ravel(), halves[2 * b + 1][i].ravel())[0, 1]:+.2f} ({np.sqrt(noise_share(halves[2 * b], 2, i) * noise_share(halves[2 * b + 1], 2, i)):.2f}) {s}'
                                           for s, i in by_source.items()) for b, a in enumerate(SQUARE)) + '.', '',
            '## The fixed persona prompt against the earlier one', '',
            'The earlier persona prompt gave the persona values and then said two more times "with your own values". The fixed prompt says "with these values" in all three places. '
            'Same questions, same model, 1 answer each.',
            'SHOULD: fewer hidden reasonings name a conflict, and a persona that was diluted moves farther from default DeepSeek. ELSE the conflicting sentence was not the cause.',
            f'r earlier to fixed: correlation of the persona preference over all options. Two answers of default DeepSeek give r = '
            + ' and '.join(f'{np.corrcoef(T[i, ds, 0].ravel(), T[i, ds, 1].ravel())[0, 1]:.2f} ({s})' for s, i in by_source.items()) + '; a lower r means that the choices changed, not only the noise.', '']
    Pp0 = load_personas(qs, mask, PERSONAS, EARLIER)
    Tp0 = p_transforms(np.nan_to_num(Pp0), mask)['logp_z'][:, :, 0]
    shared = {s: i[np.isfinite(Pp0[i, :, 0, 0]).all(1)] for s, i in by_source.items()}
    rows = [{'persona': pn, 'hidden reasoning names a conflict, earlier': names_conflict(EARLIER, pn), 'fixed': names_conflict(TAG, pn),
             **{f'{s}: {name}': f(i) for s, i in shared.items() for name, f in (
                 ('distance to default, earlier', lambda i: rms(T[:, ds, 0], Tp0[:, j], i)), ('fixed', lambda i: rms(T[:, ds, 0], Tp[:, j], i)),
                 ('r earlier to fixed', lambda i: np.corrcoef(Tp0[i, j].ravel(), Tp[i, j].ravel())[0, 1]))}} for j, pn in enumerate(PERSONAS)]
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.2f'), '',
            'Keys and scores. r of the key: correlation of the earlier and the fixed key over all options. r if only answer noise differs: the value for two draws of one unchanged key '
            '(square root of the product of the two shares that are not answer noise). r of the six scores: do the default models keep their order on the key?',
            'SHOULD: a lean that is about the models, not about one prompt wording, keeps its sign and the six scores correlate. ELSE the earlier lean came from the diluted personas.', '']
    keys0 = persona_keys(Tp0, KEYS, PERSONAS)
    rows = []
    for kname in KEYS:
        old = {s: slope(dev[i].mean(2), keys0[kname][i]) for s, i in shared.items()}
        rows.append({'key': kname, **{f'{s}: {name}': v for s, i in shared.items() for name, v in (
            ('r of the key', f'{np.corrcoef(keys0[kname][i].ravel(), keys[kname][i].ravel())[0, 1]:+.2f}'),
            ('r if only answer noise differs', f'{np.sqrt(noise_share(keys0[kname], sides(kname), i) * noise_share(keys[kname], sides(kname), i)):.2f}'),
            ('r of the six scores', f'{np.corrcoef(old[s], res[kname][s]["est"])[0, 1]:+.2f}'),
            ('range of the six scores, earlier to fixed', f'{np.ptp(old[s]):.3f} to {np.ptp(res[kname][s]["est"]):.3f}'),
            ('largest score, earlier to fixed', f'{names[np.abs(old[s]).argmax()]} {old[s][np.abs(old[s]).argmax()]:+.2f} to '
             f'{names[np.abs(res[kname][s]["est"]).argmax()]} {res[kname][s]["est"][np.abs(res[kname][s]["est"]).argmax()]:+.2f}'))}})
    out += [tabulate(rows, headers='keys', tablefmt='pipe', disable_numparse=True), '',
            f'Plots: {OUT / "persona_pairs.png"}', f'{OUT / "alignment_square.png"}', f'This file: {OUT / "run.md"}']
    (OUT / 'run.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
