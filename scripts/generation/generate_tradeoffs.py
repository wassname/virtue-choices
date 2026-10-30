"""Generate source-backed summaries and separately marked synthetic trade-offs. -- PI/OpenAI"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import string
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from tradeoff_budget import run_batch

MODEL = 'deepseek/deepseek-v4.1-flash'
PROMPT = ROOT / 'data/probes/tradeoff_generation_prompt.md'
PERSONS = {
    'first': dict(subject='I', Subject='I', object='me', possessive='my', Possessive='My', reflexive='myself', be='am'),
    'second': dict(subject='you', Subject='You', object='you', possessive='your', Possessive='Your', reflexive='yourself', be='are'),
}


def template_fields(text: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}


def validate(family: dict, source: dict) -> None:
    assert set(family) == {'source_summary', 'source_consequences', 'source_issues', 'experiment', 'generation_note'}, 'Top-level fields'
    assert isinstance(family['source_summary'], str) and family['source_summary'].strip()
    assert isinstance(family['generation_note'], str)
    consequences = family['source_consequences']
    assert len(consequences) == len(source['displayed_choices']), 'Every original option needs consequences'
    assert {str(c['choice_id']) for c in consequences} == set(source['displayed_choices'])
    for c in consequences:
        assert c['evidence'], ('Missing evidence', c['choice_id'])
        for e in c['evidence']:
            assert e['field'] in {'obs', 'choice_obs'}, e
            original = source['obs'] if e['field'] == 'obs' else source['choice_obs'][str(e['choice_id'])]
            assert isinstance(e['quote'], str) and e['quote'].strip() and e['quote'] in original, ('Non-verbatim quote', e)
            if e['field'] == 'choice_obs':
                assert str(e['choice_id']) == str(c['choice_id']), 'Evidence from a different option'
    experiment = family['experiment']
    assert experiment is not None, ('No coherent experiment', family['source_issues'])
    for key in ('question', 'target_policy', 'changed_from_source', 'shared_facts'):
        assert experiment[key], key
    values = experiment['variable']['values']
    assert len(values) == 5 and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    assert all(a < b for a, b in zip(values, values[1:])), values
    assert set(experiment['positions']) == {'original', 'opposing'}
    assert isinstance(experiment['shared_facts'], list) and all(isinstance(x, str) for x in experiment['shared_facts'])
    assert not re.search(r'target policy|five levels|variable cost', ' '.join(experiment['shared_facts']), re.I), 'Experiment metadata in actor-visible facts'
    for role, position in experiment['positions'].items():
        assert position['actor'] and position['position_note'], role
        assert 'cost' not in template_fields(position['context_template']), 'Cost must appear only in outcomes'
        assert 150 <= len(position['context_template'].split()) <= 500, ('Context word count', role, len(position['context_template'].split()))
        actions = position['actions']
        assert 2 <= len(actions) <= 6
        assert len({str(a['id']) for a in actions}) == len(actions)
        assert all(type(a['is_target_policy']) is bool for a in actions)
        assert {a['is_target_policy'] for a in actions} == {False, True}
        assert any('cost' in template_fields(a['outcome_template']) for a in actions), 'No cost manipulation'
        for a in actions:
            assert set(map(str, a['source_choice_ids'])) <= set(source['displayed_choices'])
            assert 'cost' not in template_fields(a['text_template'])
        templates = [position['context_template']] + [a[k] for a in actions for k in ('text_template', 'outcome_template')]
        assert any(template_fields(text) & set(PERSONS['first']) for text in templates), 'No grammatical-person manipulation'
        for text in templates:
            assert text.strip(), 'Empty template'
            assert template_fields(text) <= set(PERSONS['first']) | {'cost'}, template_fields(text)
            for person in PERSONS.values():
                text.format(**person, cost=values[0])


def main(inputs: Path, out: Path, offset: int, limit: int, repair_receipts: Path | None, feedback_path: Path | None) -> None:
    sources = [json.loads(line) for line in inputs.open()][offset:offset + limit]
    if repair_receipts is not None:
        feedback = json.loads(feedback_path.read_text())
        previous = {r['metadata']['source_id']: r for r in map(json.loads, repair_receipts.open())}
        sources = [s for s in sources if s['source_id'] in feedback]
        assert {s['source_id'] for s in sources} == set(feedback)
    settings = json.loads((ROOT / 'data/models/probability_eval_isekai_v12.json').read_text())['models'][MODEL]
    packets = [{'metadata': {'source_id': s['source_id'], 'stage': 'generation'}, 'request': {
        'model': MODEL, 'provider': settings['provider'], 'reasoning': {'max_tokens': 6000, 'exclude': False},
        'temperature': 0.3, 'max_tokens': 20000, 'response_format': {'type': 'json_object'},
        'messages': [{'role': 'system', 'content': PROMPT.read_text()}, {'role': 'user', 'content': json.dumps(s, ensure_ascii=False)}]}}
        for s in sources]
    if repair_receipts is not None:
        for packet in packets:
            sid = packet['metadata']['source_id']
            packet['metadata']['repair_of'] = previous[sid]['request_key']
            packet['request']['messages'] += [
                {'role': 'assistant', 'content': previous[sid]['response']['choices'][0]['message']['content']},
                {'role': 'user', 'content': 'Repair the complete JSON object using this review. Preserve correct source quotes. Do not optimize for model disagreement. Review: ' + json.dumps(feedback[sid])}]
    reservation = max((len(json.dumps(p['request']).encode()) * settings['input_usd_per_million'] +
                       p['request']['max_tokens'] * settings['output_usd_per_million']) / 1e6 for p in packets) * 1.1
    run_batch(packets, out, max_usd=1.0, reservation=reservation, concurrency=4)
    (out / 'provenance.json').write_text(json.dumps({'inputs': str(inputs.resolve()),
        'inputs_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    by_id = {s['source_id']: s for s in sources}
    families, errors = [], []
    for line in (out / 'receipts.jsonl').open():
        record = json.loads(line)
        sid = record['metadata']['source_id']
        try:
            assert record['http_status'] == 200 and 'error' not in record['response'], record['response']
            choice = record['response']['choices'][0]
            assert choice['finish_reason'] == 'stop', choice['finish_reason']
            family = json.loads(choice['message']['content'])
            validate(family, by_id[sid])
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            errors.append({'source_id': sid, 'error': str(error), 'request_key': record['request_key']})
            continue
        families.append({'source_id': sid, 'request_key': record['request_key'], **family})
    (out / 'families.jsonl').write_text(''.join(json.dumps(f, ensure_ascii=False) + '\n' for f in families))
    (out / 'validation_errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    print(f'Validated {len(families)}/{len(sources)} families; semantic review still required. {out}')
    assert not errors, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/interim/tradeoff12_sources_v2/generation_inputs.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--repair-receipts', type=Path)
    parser.add_argument('--feedback', type=Path)
    args = parser.parse_args()
    assert (args.repair_receipts is None) == (args.feedback is None)
    main(args.inputs, args.out, args.offset, args.limit, args.repair_receipts, args.feedback)
