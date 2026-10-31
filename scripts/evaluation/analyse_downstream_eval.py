"""Compare frozen downstream choices with centered labels and paired uncertainty. -- PI/OpenAI"""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from repair_downstream_replies import repaired_rows
from run_downstream_eval import CONDITIONS, ROOT, read_rows, sha
from score_choice_dimensions import bootstrap


def centered_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    assert np.allclose(probabilities.sum(-1), 1)
    assert np.isfinite(probabilities).all() and np.isfinite(labels).all()
    centered = labels - labels.mean(axis=-2, keepdims=True)
    return np.einsum('ra,lad->rld', probabilities, centered)


def interval(draws: np.ndarray) -> list:
    return np.quantile(draws, [.025, .975], axis=0).tolist()


def analyse(root: Path, replicates: int) -> dict:
    prepared = root / 'preflight_v5'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    for name, expected in manifest['frozen_sha256'].items():
        assert sha(prepared / name) == expected, name
    models = manifest['models']
    scenes = read_rows(prepared / 'scenes.jsonl')
    by_id = {s['source_id']: s for s in scenes}
    families = [s for s in scenes if s['condition'] == 'none']
    fids = [s['family_id'] for s in families]
    groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
    rubric = json.loads((ROOT / 'rubrics/virtues_v2.json').read_text())
    items = sorted(i['item_id'] for i in rubric['items'])
    assert len(items) == 66
    kinds = {i['item_id']: i['kind'] for i in rubric['items']}
    assert all(kinds[m] == 'virtues' for members in groups.values() for m in members)
    dimensions = items + list(groups)
    group_indices = list(range(len(items), len(dimensions)))
    stages = [root / ('actors_' + part + '_v5') for part in ('smoke', 'rest')]
    rows = []
    provenance = {'manifest': sha(prepared / 'manifest.json'), 'groups': sha(ROOT / 'rubrics/virtue_display_groups.json')}
    for stage in stages:
        for batch in [stage, *sorted((stage / 'repairs').glob('attempt_*'))]:
            assert (batch / 'feedback_review.md').exists(), f'Read feedback first: {batch}'
            provenance[str(batch / 'parsed.jsonl')] = sha(batch / 'parsed.jsonl')
        rows.extend(repaired_rows(stage))
    key = lambda r: (r['source_id'], r['model'], r['draw'])
    assert len({key(r) for r in rows}) == len(rows)
    expected = {key(p['metadata']) for p in read_rows(prepared / 'packets.jsonl')}
    assert {key(r) for r in rows} == expected
    assert all(r['status'] == 'ok' for r in rows), 'Missing/malformed replies cannot be scored'
    labels, dnd = {}, {}
    for part in ('smoke', 'rest'):
        stage = root / ('labels_' + part + '_v5')
        assert (stage / 'feedback_review.md').exists(), f'Read judge feedback first: {stage}'
        for condition in CONDITIONS:
            for kind, target in (('virtues', labels), ('dnd', dnd)):
                path = stage / condition / kind / 'labels.jsonl'
                provenance[str(path)] = sha(path)
                for row in read_rows(path):
                    if kind == 'virtues' and row['item_id'] not in items:
                        continue
                    k = row['source_id'], row['choice_id'], row['item_id']
                    assert k not in target, k
                    target[k] = row
    required = {(s['source_id'], a, item) for s in scenes for a in s['bare_choice_text'] for item in items}
    assert set(labels) == required
    assert set(dnd) == {(s['source_id'], a, axis) for s in scenes for a in s['bare_choice_text'] for axis in ('order', 'morality')}
    cells = defaultdict(list)
    for row in rows:
        cells[row['family_id'], row['model'], row['condition']].append(row)
    modes = ('stated', 'chosen')
    shape = (len(families), len(models), 3, 3, len(dimensions))
    means = {mode: np.empty(shape) for mode in modes}
    mean_variance = {mode: np.empty(shape) for mode in modes}
    repeat_boot = {mode: np.zeros((replicates, len(models), 3, 3, len(groups))) for mode in modes}
    null_boot = {mode: np.zeros_like(repeat_boot[mode]) for mode in modes}
    rng = np.random.default_rng(20261031)
    permutation_rng = np.random.default_rng(20261101)
    trial_scores, dnd_rows = [], []
    for fi, family in enumerate(families):
        fid = family['family_id']
        actions = list(family['bare_choice_text'])
        label_array = []
        for condition in CONDITIONS:
            sid = fid + '::' + condition
            raw = np.array([[labels[sid, a, item]['signed'] * (-1 if kinds[item] == 'vices' else 1)
                             for item in items] for a in actions])
            display = np.stack([raw[:, [items.index(item) for item in members]].mean(1) for members in groups.values()], 1)
            label_array.append(np.concatenate([raw, display], 1))
            for action in actions:
                for axis in ('order', 'morality'):
                    dnd_rows.append(dnd[sid, action, axis])
        label_array = np.array(label_array)
        uniform = np.full((1, len(actions)), 1 / len(actions))
        assert np.allclose(centered_scores(uniform, label_array), 0), 'Uniform-choice null must be zero'
        family_trials = {mode: {} for mode in modes}
        for mi, model in enumerate(models):
            for ci, condition in enumerate(CONDITIONS):
                rs = sorted(cells[fid, model, condition], key=lambda r: r['draw'])
                n = manifest['draws_per_family'][fid]
                assert [r['draw'] for r in rs] == list(range(n))
                assert all(set(r['original_probabilities']) == set(actions) for r in rs)
                probabilities = np.array([[r['original_probabilities'][a] / 100 for a in actions] for r in rs])
                chosen = np.array([[float(r['original_choice_id'] == a) for a in actions] for r in rs])
                resampled = rng.integers(0, n, size=(replicates, n))
                for mode, p in zip(modes, (probabilities, chosen), strict=True):
                    scores = centered_scores(p, label_array)
                    family_trials[mode][mi, ci] = scores[:, :, group_indices]
                    means[mode][fi, mi, ci] = scores.mean(0)
                    mean_variance[mode][fi, mi, ci] = scores.var(0, ddof=1) / n
                    repeat_boot[mode][:, mi, ci] += scores[:, :, group_indices][resampled].mean(1) / len(families)
                    for ri, row in enumerate(rs):
                        trial_scores.append({'family_id': fid, 'request_key': row['request_key'], 'model': model,
                                             'choice_condition': condition, 'mode': mode, 'scores_by_label_condition': scores[ri].tolist()})
        for ci in range(3):
            for draw in range(manifest['draws_per_family'][fid]):
                order = permutation_rng.permuted(np.broadcast_to(np.arange(len(models)), (replicates, len(models))), axis=1)
                for mode in modes:
                    scored = np.stack([family_trials[mode][mi, ci][draw] for mi in range(len(models))])
                    null_boot[mode][:, :, ci] += scored[order] / manifest['draws_per_family'][fid] / len(families)
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    (out / 'trial_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in trial_scores))
    (out / 'dnd_raw.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in dnd_rows))
    result = {'author': 'PI/OpenAI', 'analysis_source_sha256': sha(Path(__file__)),
              'models': models, 'conditions': list(CONDITIONS), 'dimensions': dimensions,
              'dimension_kinds': kinds | {g: 'virtue_group' for g in groups},
              'display_groups': list(groups), 'families': fids, 'game_clusters': [s['game_id'] for s in families],
              'dev_roles': [s['dev_role'] for s in families], 'provenance': provenance, 'replicates': replicates,
              'estimator': 'Named-trait polarity, within-question additive centering, then choice expectation, draw mean, equal-scene mean. No SD/range scaling.',
              'uncertainty': 'Paired game-cluster percentile intervals with fixed labels; repeat-only resampling within fixed scene/model/choice-condition preserves label-condition pairing. Pointwise, not simultaneous.',
              'limitations': ['Selected development scenes, not a random population sample.', 'Game text until next decision has unequal horizons.',
                              'Model-specific comprehension errors can create or hide apparent value differences.',
                              'Valid-format redraws condition sampling on format success; originals retained.',
                              'Native label uncertainty and judge-version bias are not included in bootstrap intervals.',
                              'Positive RMS bootstrap bounds do not test zero separation; RMS intervals are descriptive.',
                              'Permutation null exchanges model identities within scene/condition/draw, preserving displayed order. It tests exchangeability, not equality of mean gaps between conditions.',
                              'Empirical repeat bootstrap variance is (n-1)/n of the unbiased plug-in variance within each cell; n is only 4-6. It includes option-order effects.'], 
              'profiles': [], 'pairs': [], 'separation': [], 'scene_deletion': [], 'subset_points': []}
    saved = {}
    for mode in modes:
        values = means[mode]
        boot = bootstrap(values.reshape(len(families), -1, len(dimensions)), result['game_clusters'], replicates)
        boot = boot.reshape(replicates, len(models), 3, 3, len(dimensions))
        point = values.mean(0)
        variance = mean_variance[mode].sum(0) / len(families) ** 2
        saved[mode + '_scene_means'] = values
        saved[mode + '_mean_variance'] = variance
        saved[mode + '_cluster_bootstrap'] = boot
        saved[mode + '_repeat_bootstrap_groups'] = repeat_boot[mode]
        saved[mode + '_permutation_null_groups'] = null_boot[mode]
        for ci, condition in enumerate(CONDITIONS):
            for li, label_condition in enumerate(CONDITIONS):
                for mi, model in enumerate(models):
                    result['profiles'].append({'mode': mode, 'model': model, 'choice_condition': condition,
                        'label_condition': label_condition, 'means': point[mi, ci, li].tolist(),
                        'cluster_ci95': interval(boot[:, mi, ci, li]),
                        'repeat_ci95_groups': interval(repeat_boot[mode][:, mi, ci, li])})
        for scoring in ('matched', 'fixed-none'):
            current = np.stack([point[:, c, c if scoring == 'matched' else 0, group_indices] for c in range(3)], 1)
            draws = np.stack([boot[:, :, c, c if scoring == 'matched' else 0][:, :, group_indices] for c in range(3)], 2)
            repeat = np.stack([repeat_boot[mode][:, :, c, c if scoring == 'matched' else 0] for c in range(3)], 2)
            var = np.stack([variance[:, c, c if scoring == 'matched' else 0, group_indices] for c in range(3)], 1)
            null = np.stack([null_boot[mode][:, :, c, c if scoring == 'matched' else 0] for c in range(3)], 2)
            assert current.shape == (len(models), 3, len(groups))
            for a, b in combinations(range(len(models)), 2):
                delta = current[a] - current[b]
                dboot = draws[:, a] - draws[:, b]
                rboot = repeat[:, a] - repeat[:, b]
                result['pairs'].append({'mode': mode, 'scoring': scoring, 'a': models[a], 'b': models[b],
                    'difference_by_condition_group': delta.tolist(), 'cluster_ci95': interval(dboot),
                    'repeat_ci95': interval(rboot), 'later_minus_none': (delta[2] - delta[0]).tolist(),
                    'later_minus_none_cluster_ci95': interval(dboot[:, 2] - dboot[:, 0]),
                    'later_minus_none_repeat_ci95': interval(rboot[:, 2] - rboot[:, 0]),
                    'later_minus_immediate': (delta[2] - delta[1]).tolist(),
                    'later_minus_immediate_cluster_ci95': interval(dboot[:, 2] - dboot[:, 1]),
                    'later_minus_immediate_repeat_ci95': interval(rboot[:, 2] - rboot[:, 1])})
            pair_square = np.stack([((current[a] - current[b]) ** 2).mean(-1) for a,b in combinations(range(len(models)),2)])
            pair_noise = np.stack([(var[a] + var[b]).mean(-1) for a,b in combinations(range(len(models)),2)])
            boot_square = np.stack([((draws[:, a] - draws[:, b]) ** 2).mean(-1) for a,b in combinations(range(len(models)),2)]).mean(0)
            null_rms = np.sqrt(np.stack([((null[:,a]-null[:,b])**2).mean(-1)
                for a,b in combinations(range(len(models)),2)]).mean(0))
            result['separation'].append({'mode': mode, 'scoring': scoring,
                'permutation_null_rms_ci95': interval(null_rms),
                'permutation_null_rms_median': np.median(null_rms,axis=0).tolist(),
                'permutation_exchangeability_p': ((1+(null_rms >= np.sqrt(pair_square.mean(0))).sum(0))/(replicates+1)).tolist(),
                'rms_model_gap': np.sqrt(pair_square.mean(0)).tolist(),
                'repeat_noise_rms_for_gap_of_means': np.sqrt(pair_noise.mean(0)).tolist(),
                'squared_gap_minus_repeat_variance': (pair_square.mean(0)-pair_noise.mean(0)).tolist(),
                'rms_gap_cluster_ci95': interval(np.sqrt(boot_square)),
                'later_minus_none_rms_gap_cluster_ci95': interval(np.sqrt(boot_square[:,2])-np.sqrt(boot_square[:,0])),
                'later_minus_immediate_rms_gap_cluster_ci95': interval(np.sqrt(boot_square[:,2])-np.sqrt(boot_square[:,1]))})
            for controls in (False, True):
                mask = np.array([s['dev_role'].startswith('control') == controls for s in families])
                if not mask.any():
                    continue
                subset = values[mask].mean(0)
                mat = np.stack([subset[:,c,c if scoring=='matched' else 0,group_indices] for c in range(3)],1)
                gap = np.sqrt(np.stack([((mat[a]-mat[b])**2).mean(-1) for a,b in combinations(range(len(models)),2)]).mean(0))
                result['subset_points'].append({'mode':mode,'scoring':scoring,'subset':'controls' if controls else 'consequence',
                    'n_scenes':int(mask.sum()),'rms_gap':gap.tolist(),'model_profiles_by_condition_group':mat.tolist()})
            for fi, family in enumerate(families):
                subset = np.delete(values,fi,axis=0).mean(0)
                mat = np.stack([subset[:,c,c if scoring=='matched' else 0,group_indices] for c in range(3)],1)
                gap = np.sqrt(np.stack([((mat[a]-mat[b])**2).mean(-1) for a,b in combinations(range(len(models)),2)]).mean(0))
                result['scene_deletion'].append({'mode':mode,'scoring':scoring,'excluded':family['original_source_id'],'rms_gap':gap.tolist()})
    result['dnd_profiles'] = []
    for axis in ('order', 'morality'):
        missing = [[sid, a] for (sid, a, item), row in dnd.items() if item == axis and row['score'] is None]
        if missing:
            result['dnd_profiles'].append({'axis': axis, 'status': 'unavailable_full_cohort', 'missing_action_labels': missing})
            continue
        axis_values = {mode: np.empty(shape[:-1] + (1,)) for mode in modes}
        unclear = []
        for fi, family in enumerate(families):
            fid = family['family_id']
            actions = list(family['bare_choice_text'])
            axis_labels = np.array([[[dnd[fid + '::' + c, a, axis]['score']] for a in actions] for c in CONDITIONS])
            unclear.append([[dnd[fid + '::' + c, a, axis]['unclear_probability'] for a in actions] for c in CONDITIONS])
            for mi, model in enumerate(models):
                for ci, condition in enumerate(CONDITIONS):
                    rs = cells[fid, model, condition]
                    probabilities = np.array([[r['original_probabilities'][a]/100 for a in actions] for r in rs])
                    chosen = np.array([[float(r['original_choice_id'] == a) for a in actions] for r in rs])
                    for mode, p in zip(modes, (probabilities, chosen), strict=True):
                        axis_values[mode][fi, mi, ci] = centered_scores(p, axis_labels).mean(0)
        for mode, values in axis_values.items():
            boot = bootstrap(values.reshape(len(families), -1, 1), result['game_clusters'], replicates)
            boot = boot.reshape(replicates, len(models), 3, 3)
            result['dnd_profiles'].append({'axis': axis, 'mode': mode, 'status': 'complete',
                'definition': 'Native numerical score conditional on known grades; unclear probability retained separately, never neutral.',
                'means': values.mean(0)[..., 0].tolist(), 'cluster_ci95': interval(boot),
                'unclear_probabilities_by_scene_condition_action': unclear})
    np.savez_compressed(out / 'scores.npz', **saved)
    (out / 'estimates.json').write_text(json.dumps(result, indent=2) + '\n')
    print(out / 'estimates.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=3000)
    args = parser.parse_args()
    analyse(args.root, args.replicates)
