"""Offline checks through the downstream renderer, request builders and parser. -- PI/OpenAI"""

import argparse
import copy
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS / 'data'), str(SCRIPTS / 'evaluation')]
from downstream_inputs import CONDITIONS, JUDGE_SCOPE, future_text, label_actions, read_rows, render_scenes
from run_downstream_eval import PROTOCOL, judge_request, judge_spec
from run_tiny_eval import build, parse


def check(prepared: Path) -> None:
    records = read_rows(prepared / 'source.jsonl')
    contexts = read_rows(prepared / 'contexts.jsonl')
    scenes = render_scenes(records, contexts)
    assert scenes == read_rows(prepared / 'scenes.jsonl')
    config = json.loads((prepared / 'config.json').read_text())
    assert config['protocol'] == PROTOCOL
    model = next(iter(config['models']))
    for i in range(0, len(scenes), 3):
        variants = scenes[i:i + 3]
        assert [r['condition'] for r in variants] == list(CONDITIONS)
        assert len({r['context_sha256'] for r in variants}) == 1
        maps = []
        for row in variants:
            n = len(row['choice_text'])
            cycle = []
            for draw in range(n):
                packet = build(row | {'source_id': row['family_id']}, draw, model, config)
                mapping = packet['metadata']['displayed_to_original']
                cycle.append(mapping)
                request = packet['request']['messages'][1]['content']
                assert request.count('<current_option>') == request.count('</current_option>') == n
                for displayed, original in mapping.items():
                    assert f'{displayed}: ' + row['choice_text'][original] in request
                reply = {'reasoning': 'Fixture response.', 'choice': 0,
                         'probabilities': {k: 100 if k == '0' else 0 for k in mapping}, 'setup_note': ''}
                receipt = {'metadata': packet['metadata'], 'request_key': 'fixture', 'cache_hit': False, 'http_status': 200,
                    'response': {'provider': config['models'][model]['provider_name'], 'model': model, 'usage': {},
                        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}]}}
                parsed = parse(receipt, config)
                assert parsed['status'] == 'ok' and parsed['original_choice_id'] == mapping['0']
            for displayed in range(n):
                assert {m[str(displayed)] for m in cycle} == set(row['choice_text'])
            maps.append(cycle)
        assert maps[0] == maps[1] == maps[2]
    for record in records:
        later = next(r for r in scenes if r['original_source_id'] == record['source_id'] and r['condition'] == 'later')
        for key in record['options']:
            assert future_text(record, key) in later['text'], 'Shared blocks must preserve complete future text'
    for action in label_actions(scenes):
        row = next(r for r in scenes if r['source_id'] == action['source_id'])
        assert action['story'] == row['text'] and action['action'] == row['bare_choice_text'][action['choice_id']]
    for kind in ('virtues', 'dnd'):
        rubric, items, request_fn, _ = judge_spec(kind)
        payload = judge_request(request_fn, rubric, items[0], label_actions(scenes[:3])[:2], full_feedback=True)
        assert JUDGE_SCOPE in json.dumps(payload, ensure_ascii=False)
        if kind == 'dnd':
            assert payload['state']['rules'][0] == payload['state']['target_scope'] == JUDGE_SCOPE
            assert 'hidden game branches, later outcomes' not in json.dumps(payload['state'])
    changed = copy.deepcopy(records)
    changed[0]['immediate']['0'] += '\nUNIQUE_FUTURE_EVIDENCE'
    modified = render_scenes(changed, contexts)
    assert modified[0] == scenes[0]
    assert 'UNIQUE_FUTURE_EVIDENCE' in modified[1]['text'] and 'UNIQUE_FUTURE_EVIDENCE' in modified[2]['text']
    secret_contexts = copy.deepcopy(contexts)
    secret_contexts[0]['game_vars'] = {'secret': 'UNIQUE_HIDDEN_STATE'}
    assert render_scenes(records, secret_contexts) == scenes
    controls = [r for r in scenes if r['condition'] == 'later' and r['dev_role'].startswith('control')]
    assert controls and all(r['later_evidence_equal_across_options'] for r in controls)
    print(f'PASS: {len(scenes)} matched inputs; complete rotations; ID round-trip; explicit current/future boundaries; '
          'shared judge evidence and scope; no baseline future leakage or hidden-state leakage; controls identical; '
          'complete future text preserved. No API calls.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prepared', type=Path)
    check(ap.parse_args().prepared)
