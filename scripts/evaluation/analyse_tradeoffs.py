"""Audit feedback, then compare paired trade-off responses by source family. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import textwrap

import numpy as np
from tabulate import tabulate


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def feedback_samples(stage: Path) -> None:
    rows = read_rows(stage / 'choices.jsonl')
    models = sorted({r['model'] for r in rows})
    text = ['# Hash-selected feedback and reasoning', 'Author: PI/OpenAI. Selection key: SHA256("tradeoff-audit:" + request_key).']
    for model in models:
        ordered = sorted((r for r in rows if r['model'] == model), key=lambda r: hashlib.sha256(('tradeoff-audit:' + r['request_key']).encode()).hexdigest())
        text += [f'## {model} — setup notes and expressed reasons']
        for r in ordered[:20]:
            text += [f"### {r['source_id']} / draw {r['draw']} / {r['request_key']}",
                     f"Status: {r['status']}", 'Setup note: ' + str(r.get('setup_note', '(no parsed note)')),
                     'Reasoning: ' + str(r.get('reasoning', r.get('content', '(no content)')))]
        hidden = []
        for r in ordered:
            message = r.get('provider_message', {})
            reasoning = message.get('reasoning')
            if not reasoning:
                reasoning = '\n'.join(d['text'] for d in message.get('reasoning_details', []) if d.get('text'))
            if reasoning:
                hidden.append((r, reasoning))
        text += [f'## {model} — hidden reasoning ({len(hidden)} available; first 10 by the same hash)']
        for r, reasoning in hidden[:10]:
            text += [f"### {r['source_id']} / draw {r['draw']} / {r['request_key']}", reasoning]
    target = stage / 'feedback_samples.md'
    target.write_text('\n\n'.join(text) + '\n')
    print(target)


def ci(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    means = values[draws].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(means, [.025, .975]))


def analyse(out: Path) -> None:
    config = json.loads((out / 'config.json').read_text())
    families = read_rows(out / 'families.jsonl')
    models = list(config['models'])
    fids = [f['source_id'] for f in families]
    stages = [out / 'smoke', out / 'rest']
    for stage in stages:
        assert (stage / 'feedback_review.md').exists(), f'Read and record feedback first: {stage}'
    rows = [r for stage in stages for r in read_rows(stage / 'choices.jsonl')]
    expected = read_rows(out / 'packets.jsonl')
    key = lambda r: (r['source_id'], r['model'], r['draw'])
    assert len({key(r) for r in rows}) == len(rows), 'Duplicate trial'
    assert {key(r) for r in rows} == {key(p['metadata']) for p in expected}, 'Missing trials'
    assert all(r['status'] == 'ok' for r in rows), 'Malformed/missing judgments must be resolved, not scored'
    cells = defaultdict(list)
    for r in rows:
        k = (r['family_id'], r['model'], r['role'], r['person'], r['cost_level'])
        cells[k].append(r)
    shape = (len(families), len(models), 2, 2, 5)
    choices, probabilities, chance = (np.empty(shape) for _ in range(3))
    table = []
    for fi, fid in enumerate(fids):
        for mi, model in enumerate(models):
            for ri, role in enumerate(('original', 'opposing')):
                for pi, person in enumerate(('first', 'second')):
                    for level in range(5):
                        rs = cells[(fid, model, role, person, level)]
                        assert rs
                        target = set(rs[0]['target_ids'])
                        index = fi, mi, ri, pi, level
                        choices[index] = np.mean([r['original_choice_id'] in target for r in rs])
                        probabilities[index] = np.mean([sum(r['original_probabilities'][k] for k in target) / 100 for r in rs])
                        chance[index] = len(target) / len(rs[0]['original_probabilities'])
                        table.append({'family_id': fid, 'model': model, 'role': role, 'person': person,
                                      'cost_level': level, 'cost': rs[0]['cost'], 'unit': rs[0]['unit'], 'n': len(rs),
                                      'chosen_target': choices[index], 'stated_target': probabilities[index], 'chance': chance[index]})
    assert len(cells) == int(np.prod(shape))
    (out / 'cell_means.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in table))
    per_family = {
        'target_choice': choices.mean(axis=(2, 3, 4)),
        'stated_probability': probabilities.mean(axis=(2, 3, 4)),
        'cost_high_minus_low': (choices[..., -1] - choices[..., 0]).mean(axis=(2, 3)),
        'first_minus_second': (choices[:, :, :, 0, :] - choices[:, :, :, 1, :]).mean(axis=(2, 3)),
        'opposing_minus_original': (choices[:, :, 1, :, :] - choices[:, :, 0, :, :]).mean(axis=(2, 3)),
    }
    boot = np.random.default_rng(20261030).integers(0, len(families), (10000, len(families)))
    summaries = []
    for mi, model in enumerate(models):
        summaries.append({'model': model, **{metric: ci(array[:, mi] * 100, boot) for metric, array in per_family.items()}})
    pairs = []
    for a, b in itertools.combinations(range(len(models)), 2):
        pairs.append({'a': models[a], 'b': models[b], **{metric: ci((array[:, a] - array[:, b]) * 100, boot)
                                                      for metric, array in per_family.items()}})
    leave_one_out = []
    for excluded, fid in enumerate(fids):
        kept = np.arange(len(fids)) != excluded
        means = {metric: array[kept].mean(axis=0) * 100 for metric, array in per_family.items()}
        leave_one_out.append({'excluded_family': fid,
            'summary': [{'model': model, **{metric: float(v[mi]) for metric, v in means.items()}}
                        for mi, model in enumerate(models)],
            'paired_model_differences': [{'a': models[a], 'b': models[b],
                **{metric: float(v[a] - v[b]) for metric, v in means.items()}}
                for a, b in itertools.combinations(range(len(models)), 2)]})
    (out / 'estimates.json').write_text(json.dumps({'summary': summaries, 'paired_model_differences': pairs,
        'leave_one_family_out_point_estimates': leave_one_out,
        'per_family': {k: v.tolist() for k, v in per_family.items()}, 'families': fids, 'models': models}, indent=2) + '\n')
    fmt = lambda v: f'{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]'
    report = ['# Controlled trade-off pilot', 'Author: PI/OpenAI.',
              f'{len(families)} selected source families, {len(rows)} replies, {len(models)} models. These are synthetic adaptations, not original game outcomes.',
              'Percentages and percentage-point differences; brackets are paired source-family bootstrap 95% intervals (10,000 resamples). Twelve selected families do not support population-wide claims. Target policies are descriptive outcomes, not a common goodness scale.',
              tabulate([[config['models'][s['model']]['display_name']] + [fmt(s[k]) for k in per_family] for s in summaries],
                  headers=['Model', 'Target chosen %', 'Stated target %', 'High−low cost (pp)', 'First−second (pp)', 'Opposing−original (pp)'], tablefmt='pipe'),
              'Chance target selection, averaged with equal family/role/person/level weights: ' + f'{chance.mean() * 100:.1f}%. Cost/person contrasts have a zero chance expectation. Role differences can have unequal chance rates and different agency.',
              'Raw cell means: cell_means.jsonl. All paired model comparisons, per-family effects and leave-one-family-out point estimates: estimates.json. Leave-one-out results test dependence on individual families, including the Superlatives wording issue flagged before full-run analysis; they are not confidence intervals. Feedback: smoke/feedback_review.md and rest/feedback_review.md. No individual family or model pair was selected for this aggregate after inspecting outcomes.']
    (out / 'results.md').write_text('\n\n'.join(report) + '\n')
    print(out / 'results.md')


def plots(out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert (out / 'results.md').exists(), 'Run audited analysis first'
    cells = read_rows(out / 'cell_means.jsonl')
    families = read_rows(out / 'families.jsonl')
    config = json.loads((out / 'config.json').read_text())
    colors = {'Muse': '#0467DF', 'Qwen': '#6950EF', 'Gemini': '#E37400', 'Luna': '#111111', 'Haiku': '#8C4A2F', 'DeepSeek': '#777777'}
    for role in ('original', 'opposing'):
        fig, axes = plt.subplots((len(families) + 2) // 3, 3, figsize=(15, 3.2 * ((len(families) + 2) // 3)), sharey=True, squeeze=False, layout='constrained')
        for fi, (family, ax) in enumerate(zip(families, axes.flat)):
            ex = family['experiment']
            for model, setting in config['models'].items():
                values = [np.mean([r['chosen_target'] for r in cells if r['family_id'] == family['source_id'] and r['model'] == model and r['role'] == role and r['cost_level'] == level]) * 100 for level in range(5)]
                name = setting['display_name']
                ax.plot(range(5), values, marker='o', markersize=3, linewidth=1.4, alpha=.85, color=colors[name], label=name)
            chance = next(r['chance'] for r in cells if r['family_id'] == family['source_id'] and r['role'] == role)
            ax.axhline(chance * 100, color='#bbbbbb', linestyle=':', linewidth=.7)
            game = family['source_id'].split('_')[1].replace('-', ' ')
            policy = ex['target_policy']
            ax.set_title(f'{fi + 1}. {game}\n' + textwrap.fill(policy if len(policy) <= 100 else policy[:97] + '...', 48), fontsize=9)
            ax.set_xticks(range(5), [str(v) for v in ex['variable']['values']], fontsize=9)
            ax.set_xlabel(textwrap.fill(ex['variable']['unit'], 45), fontsize=9)
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 50, 100])
            ax.spines[['top', 'right']].set_visible(False)
            if fi % 3 == 0:
                ax.set_ylabel('Target policy chosen (%)')
        for ax in list(axes.flat)[len(families):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), frameon=False)
        fig.suptitle(f'{role.capitalize()} actor: increasing stipulated cost\nEach point averages both grammatical persons; cost levels equally spaced. Dotted line: uniform random choice.', fontsize=12)
        for ext in ('png', 'svg'):
            fig.savefig(out / f'cost_curves_{role}.{ext}', dpi=160)
        plt.close(fig)
    print(out / 'cost_curves_original.png')
    print(out / 'cost_curves_opposing.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['samples', 'analyse', 'plots'])
    p.add_argument('path', type=Path)
    args = p.parse_args()
    if args.command == 'samples':
        feedback_samples(args.path)
    elif args.command == 'analyse':
        analyse(args.path)
    else:
        plots(args.path)
