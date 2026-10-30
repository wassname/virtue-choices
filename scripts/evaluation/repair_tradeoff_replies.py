"""Redraw malformed trade-off replies without changing prompts or valid trials. -- PI/OpenAI"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from analyse_tradeoffs import load_trials, read_rows
from eval_api import cache_key
from run_tiny_eval import parse
from tradeoff_budget import run_batch


def repair(out: Path, attempt: int) -> None:
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest['frozen_sha256'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected, name
    rows, stages = load_trials(out)
    bad = [r for r in rows if r['status'] != 'ok']
    assert bad and all(r['status'] == 'probability_sum_error' for r in bad), 'Review other failure types first'
    assert attempt == len(stages) - 1, 'Repair attempts must be consecutive'
    original = {(p['metadata']['source_id'], p['metadata']['model'], p['metadata']['draw']): p
                for p in read_rows(out / 'packets.jsonl')}
    packets = []
    for row in bad:
        packet = copy.deepcopy(original[(row['source_id'], row['model'], row['draw'])])
        packet['metadata']['replaces_request_key'] = row['request_key']
        packet['metadata']['original_request_key'] = cache_key(packet)
        packet['cache_scope']['format_repair_attempt'] = attempt
        packets.append(packet)
    batch = out / 'repairs' / f'attempt_{attempt:02d}'
    run_batch(packets, batch, max_usd=1, reservation=manifest['reservation_usd'], concurrency=6)
    config = json.loads((out / 'config.json').read_text())
    replies = [parse(r, config) for r in read_rows(batch / 'receipts.jsonl')]
    (batch / 'choices.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in replies))
    print({status: sum(r['status'] == status for r in replies) for status in sorted({r['status'] for r in replies})})
    assert all(r['status'] == 'ok' for r in replies), 'Malformed redraws retained; inspect before another attempt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    repair(args.out, args.attempt)
