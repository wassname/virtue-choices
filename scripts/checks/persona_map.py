"""Default models and personas on one map, and how few questions the map needs.

Each axis is a persona key (see persona_keying.py) from one opposed pair: x = utilitarian minus rule-follower, y = selfish minus loyal.
Coordinate of a subject = slope of (its preference minus the six-model mean) on the key, so the two personas of an axis are 1.0 apart.
Writes persona_map.png and persona_map.md to outputs/20261110_diagnosis.
Run from scripts/checks: uv run --locked --no-sync python persona_map.py -- Claudypoo[fable], 2026-11-10"""

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from tabulate import tabulate

from normalisation_sweep import RUN, dim_sets, load, p_transforms, stat
from persona_keying import BAND, COLOUR, DISPLAY, INK, MARK, MUTED, OUT, PERSONAS, SOURCES, SURFACE, load_personas, persona_keys, plt, score_source, slope
from variance_budget import profile, split

AXES = {'outcome over rule': (['utilitarian'], ['rule-follower']), 'self over loyalty': (['selfish'], ['loyal'])}
AXIS_LABEL = ('← like rule-follower        like utilitarian →', '← like loyal        like selfish →')
X, Y = AXES
N_KEEP = (50, 100, 200)
ZOOM = .25


def label_points(ax, xs, ys, labels):
    """Names in a column at the left or the right edge, in the order of y, each joined to its point by a thin line."""
    left = xs < np.median(xs)
    for side, sel in ((-1, left), (1, ~left)):
        order = np.flatnonzero(sel)[np.argsort(ys[sel])]
        slots = ys[order].mean() + (np.arange(len(order)) - (len(order) - 1) / 2) * ZOOM * .22
        for i, slot in zip(order, slots):
            ax.annotate(labels[i], (xs[i], ys[i]), xytext=(side * ZOOM * .95, slot), ha='left' if side < 0 else 'right', va='center', fontsize=10,
                        bbox=dict(fc=SURFACE, ec='none', pad=1), arrowprops=dict(arrowstyle='-', color=MUTED, lw=.5, shrinkB=5))


def plot(pos, names, band, not_drawn):
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (whole, *zooms) = plt.subplots(1, 3, figsize=(15, 7), facecolor=SURFACE)
    whole.add_patch(Rectangle((-ZOOM, -ZOOM), 2 * ZOOM, 2 * ZOOM, fill=False, ec=MUTED, ls=(0, (4, 3)), lw=.9))
    whole.annotate('the six default models\n(enlarged on the right)', (-ZOOM, ZOOM), xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=9)
    ds = names.index('DeepSeek')
    for (sname, zoom) in zip(SOURCES.values(), zooms):
        x, y, p = pos[sname]['model'][X], pos[sname]['model'][Y], pos[sname]['persona']
        for j, pn in enumerate(PERSONAS):
            whole.plot([x['est'][ds], p[j, 0]], [y['est'][ds], p[j, 1]], color=COLOUR[sname], lw=.8, alpha=.45)
            whole.plot(*p[j], **{**MARK[sname], 'marker': 'D'})
        whole.plot(x['est'], y['est'], ls='', **{**MARK[sname], 'ms': 4, 'mew': 1})
        zoom.add_patch(Rectangle((-band, -band), 2 * band, 2 * band, color=BAND, lw=0, zorder=0))
        for i in range(len(names)):
            zoom.plot([x['lo'][i], x['hi'][i]], [y['est'][i]] * 2, color=COLOUR[sname], lw=1.2)
            zoom.plot([x['est'][i]] * 2, [y['lo'][i], y['hi'][i]], color=COLOUR[sname], lw=1.2)
            zoom.plot(x['est'][i], y['est'][i], **MARK[sname])
        label_points(zoom, x['est'], y['est'], names)
        zoom.set_xlim(-ZOOM, ZOOM); zoom.set_ylim(-ZOOM, ZOOM)
        zoom.set_title(f'Enlarged: the six default models, {sname}', fontsize=11, loc='left')
    twins = np.stack([pos[s]['persona'] for s in pos])  # (source, persona, xy): one label for the two points of a persona
    for j, pn in enumerate(PERSONAS):
        below = twins[:, j, 1].mean() < -ZOOM
        whole.annotate(pn, (twins[:, j, 0].mean(), twins[:, j, 1].min() if below else twins[:, j, 1].max()), xytext=(0, -11 if below else 11),
                       textcoords='offset points', ha='center', va='top' if below else 'bottom', fontsize=10)
    whole.set_xlim(-.8, .8); whole.set_ylim(-.6, 1)
    whole.set_title('Whole space: a persona prompt moves DeepSeek far', fontsize=11, loc='left')
    for ax in (whole, *zooms):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, lw=.8); ax.axvline(0, color=MUTED, lw=.8)
        ax.set_aspect('equal')
        ax.set_xlabel(AXIS_LABEL[0]); ax.set_ylabel(AXIS_LABEL[1])
        ax.tick_params(length=0)
        ax.grid(color=BAND, lw=.6); ax.set_axisbelow(True)
        for side in ax.spines.values():
            side.set_visible(False)
    handles = [Line2D([], [], ls='', label=f'default model, {s}', **MARK[s]) for s in SOURCES.values()]
    handles += [Line2D([], [], ls='', label=f'DeepSeek with a persona prompt, {s}', **{**MARK[s], 'marker': 'D'}) for s in SOURCES.values()]
    handles.append(Patch(color=BAND, label='no real signal (2 sd)'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.005, .95), ncols=5, frameon=False, fontsize=9)
    fig.suptitle('The six default models are close together on these two persona axes, but not over all directions (see the last line)', x=.01, ha='left', fontsize=12)
    fig.text(.01, .012, '0 = the mean of the six default models. The two personas of an axis are 1.0 apart on it by definition; the map shows where the default models '
             'fall between them.\nLines join DeepSeek to its own personas (only DeepSeek got the persona prompts; one answer per question). '
             'Crosses are 95% intervals (games or topics resampled).\nGrey square: range of positions when the persona answers come from unrelated scenes.\n' + not_drawn,
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .92))
    fig.savefig(OUT / 'persona_map.png', dpi=160, facecolor=SURFACE)


def main():
    qs, dims, models, P, mask, labels, src, explore = load()
    names = [next(v for k, v in DISPLAY.items() if k in m) for m in models]
    gem, ds = names.index('Gemini'), names.index('DeepSeek')
    Pp = load_personas(qs, mask)
    ok = np.isfinite(Pp[:, :, 0, 0]).all(1)
    T, Tp = p_transforms(P, mask)['logp_z'], p_transforms(np.nan_to_num(Pp), mask)['logp_z'][:, :, 0]
    dev = T - T.mean(1, keepdims=True)
    devP = Tp - T.mean((1, 2))[:, None]  # personas against the same origin as the models
    cl = np.array([q['eval_cluster'] for q in qs])
    n_opts = mask.sum(1)
    keys = persona_keys(Tp, AXES)
    score = lambda key, idx: score_source(dev, key, idx, cl, explore, n_opts)
    cell = lambda r, m: f'{r["est"][m]:+.3f} [{r["lo"][m]:+.2f}, {r["hi"][m]:+.2f}]'
    by_source = {sname: np.flatnonzero((src == s) & ok) for s, sname in SOURCES.items()}

    rms = lambda a, b, idx: np.sqrt(((a[idx] - b[idx]) ** 2).sum() / mask[idx].sum())
    pairs = list(combinations(range(len(names)), 2))
    dist = [{'pair': 'the same model, answer 1 against answer 2 (noise only)', **{s: np.mean([rms(T[:, m, 0], T[:, m, 1], i) for m in range(len(names))]) for s, i in by_source.items()}},
            {'pair': 'two default models (mean of 15 pairs)', **{s: np.mean([rms(T[:, a, 0], T[:, b, 1], i) for a, b in pairs]) for s, i in by_source.items()}}]
    dist += [{'pair': f'DeepSeek against DeepSeek with the {pn} prompt', **{s: rms(T[:, ds, 0], Tp[:, j], i) for s, i in by_source.items()}} for j, pn in enumerate(PERSONAS)]
    dist += [{'pair': f'{PERSONAS[a]} against {PERSONAS[b]}', **{s: rms(Tp[:, a], Tp[:, b], i) for s, i in by_source.items()}} for a, b in combinations(range(len(PERSONAS)), 2)]
    not_drawn = ('Not drawn: over all directions together, two default models differ about as much as DeepSeek and its utilitarian persona\n(root mean square difference '
                 f'of preference per option, games: {dist[1]["games"]:.2f} for the mean of the 15 pairs of models against {dist[2]["games"]:.2f}; two answers of one model: {dist[0]["games"]:.2f}).')
    pos = {sname: {'model': {a: score(keys[a], idx) for a in AXES}, 'persona': np.stack([slope(devP[idx], keys[a][idx]) for a in AXES], 1)} for sname, idx in by_source.items()}
    band = 2 * max(pos[s]['model'][a]['null_sd'].max() for s in pos for a in AXES)
    OUT.mkdir(parents=True, exist_ok=True)
    plot(pos, names, band, not_drawn)

    out = ['# Default models and personas on one map', '',
           f'Run: `{Path(__file__).name}`; {int(ok.sum())} questions that all four personas answered; preference scale `logp_z` '
           '(log of the stated percentage, centred over options, unit spread per answer). -- Claudypoo[fable]', '',
           f'x = {X} (utilitarian minus rule-follower). y = {Y} (selfish minus loyal). 0 = the mean of the six default models. '
           'Personas are DeepSeek with a persona prompt, 1 answer.', '',
           'SHOULD: the two personas of an axis are 1.0 apart on that axis (by definition), and the default models are much closer to 0 than those personas. '
           'ELSE the map does not separate "what a prompt can do" from "how the default models differ".', '']
    rows = []
    for sname in SOURCES.values():
        x, y, p = pos[sname]['model'][X], pos[sname]['model'][Y], pos[sname]['persona']
        rows += [{'source': sname, 'subject': f'persona: {pn}', 'x': f'{p[j, 0]:+.3f}', 'y': f'{p[j, 1]:+.3f}', 'distance from 0': np.hypot(*p[j])} for j, pn in enumerate(PERSONAS)]
        rows += [{'source': sname, 'subject': n, 'x': cell(x, i), 'y': cell(y, i), 'distance from 0': np.hypot(x['est'][i], y['est'][i])} for i, n in enumerate(names)]
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.2f'), '', f'Control band on the plot: +-{band:.3f} (2 sd of the scene-shuffled control).',
            'A persona carries its own answer noise into the key that it defines, so the scores of the default models are a little too small (lower bound).', '',
            '## Distance without a choice of axes', '',
            'Root mean square difference of preference per option between two subjects, 1 answer each, so answer noise is in every row.',
            'SHOULD: if the default models are "all the same", the row for two default models is near the noise row and far below the persona rows. '
            'ELSE the default models differ about as much as a persona prompt changes DeepSeek, in directions that the map does not draw.', '']

    out += [tabulate(dist, headers='keys', tablefmt='pipe', floatfmt='.2f'), '',
            '## Which questions to keep', '',
            'SHOULD: if a selection rule finds questions that show stable virtue profiles, the profile share (part of the model-specific repeatable spread that is '
            'the same in all scenes) goes up. ELSE the rule only finds questions where the models disagree scene by scene.', '']

    S = stat(T, dim_sets(dims, labels, mask, src)[0]['virt66'], mask, 'cov')
    conflict = dict(pl.read_parquet(RUN / 'conflict_filter.parquet').select('source_id', 'conflict').iter_rows())
    conflict = np.array([conflict[q['source_id']] for q in qs])
    spread = ((Tp - Tp.mean(1, keepdims=True)) ** 2).sum((1, 2))  # how far the four personas disagree on a question
    energy = (keys[X] ** 2).sum(1)
    rows, few = [], []
    for sname, idx in by_source.items():
        rules = {'all questions': idx, 'label rule: no option is best on every virtue group (conflict_filter.py, PI/OpenAI)': idx[conflict[idx]],
                 'persona rule: the 100 questions where the four personas disagree most': idx[np.argsort(-spread[idx])[:100]]}
        for rule, i in rules.items():
            r, sh = score(keys[X], i), split(T[i][:, :, 0], T[i][:, :, 1])
            rows.append({'source': sname, 'rule': rule, 'questions': len(i), 'same for all models': sh[0], 'model-specific, repeatable': sh[1], 'answer noise': sh[2],
                         'profile share, 66 virtues': profile(S, i, cl)[0], f'Gemini, {X} [95%]': cell(r, gem), 'control band': f'+-{2 * r["null_sd"].max():.3f}'})
        for n in (*N_KEEP, len(idx)):
            i = idx[np.argsort(-energy[idx])[:n]]
            r = score(keys[X], i)
            few.append({'source': sname, 'questions kept': n, 'share of the key kept': energy[i].sum() / energy[idx].sum(), 'Gemini, 2 answers [95%]': cell(r, gem),
                        'Gemini, answer 1 only': f'{slope(dev[i][:, :, 0], keys[X][i])[gem]:+.3f}',
                        'largest other model': f'{np.abs(np.delete(r["est"], gem)).max():.3f}', 'control band': f'+-{2 * r["null_sd"].max():.3f}'})
    out += [tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.2f'), '',
            f'## How few questions the "{X}" score needs', '',
            'Questions are kept in order of the size of the key (utilitarian and rule-follower personas disagree most). This rule uses persona answers only, '
            'not the answers of the default models.',
            'SHOULD: the Gemini score stays the same and its interval stays outside the control band when questions with a small key are dropped. ELSE the full set is needed.', '',
            tabulate(few, headers='keys', tablefmt='pipe', floatfmt='.2f'), '',
            f'Plot: {OUT / "persona_map.png"}', f'This file: {OUT / "persona_map.md"}']
    (OUT / 'persona_map.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
