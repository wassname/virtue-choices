"""Redraw malformed downstream replies, preserving payloads and links. -- PI/OpenAI"""

import argparse
import asyncio
import copy
import fcntl
import json
from pathlib import Path

from eval_api import cache_key, run_requests
from execute_downstream_eval import BUDGET, ROOT, conservative_total
from run_downstream_eval import read_rows, sha, write_rows
from run_tiny_eval import parse


def repaired_rows(run: Path) -> list[dict]:
    rows = {r['request_key']: r for r in read_rows(run / 'parsed.jsonl')}
    assert len(rows) == len(read_rows(run / 'packets.jsonl'))
    for stage in sorted((run / 'repairs').glob('attempt_*')):
        packets = {cache_key(p): p for p in read_rows(stage / 'packets.jsonl')}
        for row in read_rows(stage / 'parsed.jsonl'):
            old = rows.pop(row['replaces_request_key'])
            assert old['status'] == 'probability_sum_error'
            for key in ('source_id', 'model', 'draw', 'displayed_to_original'):
                assert row[key] == old[key], key
            assert row['request_key'] in packets and row['request_key'] not in rows
            rows[row['request_key']] = row
    return list(rows.values())


def repair(prepared: Path, run: Path, attempt: int) -> None:
    budget = json.loads(BUDGET.read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    assert sha(prepared / 'manifest.json') == budget['preflight_sha256']
    for name, expected in manifest['frozen_sha256'].items():
        assert sha(prepared / name) == expected, name
    assert json.loads((run / 'budget.json').read_text())['preflight_sha256'] == budget['preflight_sha256']
    assert attempt == len(list((run / 'repairs').glob('attempt_*'))) + 1
    bad = [r for r in repaired_rows(run) if r['status'] != 'ok']
    assert bad and all(r['status'] == 'probability_sum_error' for r in bad)
    originals = {(p['metadata']['source_id'], p['metadata']['model'], p['metadata']['draw']): p
                 for p in read_rows(run / 'packets.jsonl')}
    frozen = {cache_key(p): p for p in read_rows(prepared / 'packets.jsonl')}
    packets = []
    for row in bad:
        original = originals[row['source_id'], row['model'], row['draw']]
        assert frozen[cache_key(original)] == original
        packet = copy.deepcopy(original)
        packet['metadata']['replaces_request_key'] = row['request_key']
        packet['metadata']['original_request_key'] = cache_key(original)
        packet['cache_scope']['format_repair_attempt'] = attempt
        assert packet['request'] == original['request']
        packets.append(packet)
    out = run / 'repairs' / f'attempt_{attempt:02d}'
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / 'packets.jsonl', packets)
    (out / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
    ceiling = budget['project_start_conservative_usd'] + budget['additional_cap_usd']
    with (ROOT / '.local/labelling.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = asyncio.run(run_requests(packets, out, max_new_usd=min(1, ceiling - conservative_total()),
            concurrency=6, key_source=budget['key_source'], project_cap_usd=ceiling,
            evaluation_cap_usd=budget['additional_cap_usd'], reservation_usd=manifest['actor_reservation_usd'],
            scope=budget['scope']))
        config = json.loads((prepared / 'config.json').read_text())
        rows = [parse(r, config) for r in read_rows(out / 'receipts.jsonl')]
        write_rows(out / 'parsed.jsonl', rows)
        assert len(rows) == len(packets)
        summary['parse_status'] = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}
        summary['allowance_used_conservative_usd'] = conservative_total() - budget['project_start_conservative_usd']
        assert summary['allowance_used_conservative_usd'] <= budget['additional_cap_usd']
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    assert all(r['status'] == 'ok' for r in rows), 'Malformed redraws retained; inspect before another attempt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    repair(args.prepared, args.run, args.attempt)
