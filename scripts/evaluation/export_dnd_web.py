"""Export aggregate chart data without private question IDs or traces. -- PI/OpenAI"""

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_probability_eval import REFERENCE
from plot_dnd_classic import ROOT, digest, estimate, pack, variance_key
from plot_probability_eval import load_calibration

PERSONAS = ('lawful-good', 'chaotic-good', 'lawful-evil', 'chaotic-evil')


def persona_points(tables, tag, ids):
    """DeepSeek with a persona system prompt, on the axes of the default DeepSeek anchor. It is the same model, so its stated percentages are used as they are. One answer per question, so no interval. -- Claudypoo[fable]"""
    lines, _, _, folds = load_calibration(tables)
    index = [next(i for i, d in enumerate(lines[0]['_meta']['dimensions']) if d['id'] == key) for key in ('order', 'morality')]
    label_fn = lambda q: np.array([[q['option_scores'][o][i] for i in index] for o in q['options']], float)

    def point(qs, answers):  # answers: source_id -> {option id: percent}; pack wants two draws, so the one answer is given twice
        data = pack([dict(q, models={REFERENCE: q['models'][REFERENCE], 'subject': {'draws': [{'draw': d, 'probabilities': answers[q['source_id']]} for d in (0, 1)]}}) for q in qs], 'subject', label_fn, folds)
        return [estimate(data['coeff'], data['den'], data['logs'][m][:, 1]) for m in ('subject', REFERENCE)]

    out = {}
    for source in ids:
        qs = [q for q in lines[1:] if q['source_id'] in ids[source]]
        assert len(qs) == len(ids[source])
        alone, anchor = point(qs, {q['source_id']: q['models'][REFERENCE]['draws'][0]['probabilities'] for q in qs})
        assert np.abs(alone - anchor).max() < .05, (source, alone, anchor)  # default DeepSeek from one of its answers stays at its anchor
        out[source] = {}
        for name in PERSONAS:
            (run,) = (ROOT / 'outputs').glob(f'*_{tag}_{name}')
            rows = [json.loads(line) for f in sorted(run.glob('*/choices.jsonl')) for line in f.open()]
            assert len(rows) == len({r['source_id'] for r in rows}) and all(r['status'] == 'ok' for r in rows if r['source_id'] in ids[source])
            out[source][name] = {'absolute': point(qs, {r['source_id']: r['original_probabilities'] for r in rows})[0].tolist()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / '.local/choice_preview/random1000/dnd_classic.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261027_random1000_v12/calibration_heldout/dnd_web.json')
    parser.add_argument('--tables', type=Path, default=ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
    parser.add_argument('--personas', default='persona-v3', help='tag of the persona runs: outputs/*_<tag>_<persona>')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data['display_interval'] == 'cluster_refit' and data['bootstrap_replicates'] == 999
    web = {'author': 'PI/OpenAI', 'axes': ['lawful', 'good'], 'unit': 'nats', 'reference': 'DeepSeek',
           'bootstrap_replicates': data['bootstrap_replicates'], 'geometry': data['geometry'],
           'intervals': {'ci95_fixed': 'Nominal marginal cluster-bootstrap percentiles, maps and observed answers fixed.',
                         'ci95_refit': 'Nominal marginal cluster-bootstrap percentiles, opposite-fold maps refitted, observed answers fixed.',
                         'ci95_fixed_answer_resampling': 'Page-style fixed-map cluster plus empirical answer resampling; separate diagnostic assumptions.'},
           'variance_key': 'Piece width = full cluster-refit interval width times diagnostic variance / sum of diagnostic variances. These overlapping one-factor weights are not independent variance contributions; the answer piece is diagnostic, not fresh-generation uncertainty in the conditional cross.',
           'sources': {}, 'source_sha256': digest(args.input)}
    for source, record in data['sources'].items():
        key = variance_key(record)
        shares = np.array(key['normalized_diagnostic_variance_weights'])
        assert np.allclose(shares.sum(0), 1, atol=1e-14) and np.all(shares >= 0)
        anchor = record['models']['DeepSeek']['absolute']
        out = {'deepseek_anchor': anchor, 'models': {}, 'key_model': record['representative_model'],
               'variance_shares': dict(zip(('questions', 'answers', 'calibration'), shares.tolist())),
               'sum_var_over_full': key['sum_variances_over_full_variance'],
               'questions': record['questions'], 'informative_questions': record['informative_questions']}
        for name, r in record['models'].items():
            if name == 'DeepSeek':
                continue
            assert np.max(abs(np.array(r['absolute']) - (np.array(anchor) + r['offset']))) < 1e-12
            u = r['uncertainty']
            out['models'][name] = {'model': r['model'], 'offset': r['offset'], 'absolute': r['absolute'],
                                   'ci95_fixed': u['clusters_only']['ci95'], 'ci95_refit': u['cluster_refit']['ci95'],
                                   'ci95_fixed_answer_resampling': u['fixed_maps']['ci95']}
            assert np.isfinite([out['models'][name][k] for k in ('offset', 'absolute')]).all()
            for k in ('ci95_fixed', 'ci95_refit', 'ci95_fixed_answer_resampling'):
                intervals = np.array(out['models'][name][k])
                assert intervals.shape == (2, 2) and np.isfinite(intervals).all() and np.all(intervals[:, 0] < intervals[:, 1])
        assert len(out['models']) == 5
        web['sources'][source] = out
    for source, points in persona_points(args.tables, args.personas, {s: set(r['question_ids']) for s, r in data['sources'].items()}).items():
        web['sources'][source]['personas'] = points
    web['caption'] = 'Models favour lawful and good options relative to an even split; region boundaries are illustrative. Insets show calibrated offsets from DeepSeek. The uncertainty key normalizes overlapping diagnostic variances.'
    day, game = (web['sources'][s] for s in ('daily_dilemmas', 'machiavelli'))
    dr, gr = day['sum_var_over_full'], game['sum_var_over_full']
    extent = web['geometry']['extent_nats']
    radius = web['geometry']['neutral_octagon_circumradius_nats']
    web['footnote'] = ('Pair-gap-weighted log-odds, DeepSeek confidence scale; nominal cluster-bootstrap intervals condition on the two observed answers. '
                       f'Extent ±{extent:g} nats; octagon radius {radius:g}. Everyday/games: {day["questions"]}/{game["questions"]} questions; key models {day["key_model"]}/{game["key_model"]}. '
                       f'Sum of diagnostic variances/full variance, lawful/good: everyday {dr[0]:.2f}/{dr[1]:.2f}, games {gr[0]:.2f}/{gr[1]:.2f}. '
                       'Weights overlap; coverage is unvalidated.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(web, indent=2, allow_nan=False) + '\n')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
