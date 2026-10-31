"""Execute frozen downstream inputs under one approved cap. -- PI/OpenAI"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from loguru import logger

from run_downstream_eval import ROOT, CONDITIONS, judge_request, judge_spec, label_actions, lv, read_rows, sha, write_rows
from run_tiny_eval import parse
from eval_api import run_requests
from billing_reservations import outstanding

BUDGET = ROOT / '.local/downstream_dev20_budget.json'
SMOKE_IDS = (
    'machiavelli_broadsides-hms-foraker_HMS_Foraker-gosub_1073-5715',
    'machiavelli_dragon_Choice_of_the_Dragon-goblinhero_15-249',
    'machiavelli_skystrike_Skystrike_Wings_of_Justice-06dam_75-715',
)


def conservative_total() -> float:
    return sum(r['cost_usd'] for r in read_rows(ROOT / '.local/billing.jsonl')) + outstanding(ROOT)


def execute(prepared: Path, out: Path, stage: str) -> None:
    budget = json.loads(BUDGET.read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    assert sha(prepared / 'manifest.json') == budget['preflight_sha256']
    for name, expected in manifest['frozen_sha256'].items():
        assert sha(prepared / name) == expected, name
    review = json.loads((prepared / 'input_review.json').read_text())
    assert review['scenes_sha256'] == sha(prepared / 'scenes.jsonl')
    assert set(review['accepted_source_ids']) == {r['source_id'] for r in read_rows(prepared / 'source.jsonl')}
    for name in ('scripts/data/downstream_inputs.py', 'scripts/evaluation/run_tiny_eval.py'):
        assert sha(ROOT / name) == manifest['code_sha256'][name], name
    smoke = stage.endswith('smoke')
    calibration = stage == 'labels-calibration-smoke'
    if stage == 'labels-smoke':
        checked = json.loads((prepared.parent / 'native_calibration_review.json').read_text())
        assert checked['approved_for_smoke'] and checked['preflight_sha256'] == sha(prepared / 'manifest.json')
    if not smoke:
        approval = json.loads((prepared.parent / 'smoke_review.json').read_text())
        assert approval['approved_for_expansion'] and approval['preflight_sha256'] == sha(prepared / 'manifest.json')
    scenes = [r for r in read_rows(prepared / 'scenes.jsonl') if (r['original_source_id'] in SMOKE_IDS) == smoke]
    assert len(scenes) == (3 if smoke else 17) * 3
    config = json.loads((prepared / 'config.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
    (out / 'execution.json').write_text(json.dumps({'author': 'PI/OpenAI', 'stage': stage,
        'prepared': str(prepared), 'started_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'smoke_selection': 'Agent-selected theft/conditional branching/shared-future control', 'smoke_ids': SMOKE_IDS,
        'code_sha256': {name: sha(ROOT / name) for name in (*manifest['code_sha256'], 'scripts/evaluation/execute_downstream_eval.py', 'scripts/data/billing_reservations.py')}
    }, indent=2) + '\n')
    logger.remove()
    logger.add(sys.stdout, format='{message}', level='INFO')
    logger.add(out / 'run.md', format='{message}', level='DEBUG')
    ceiling = budget['project_start_conservative_usd'] + budget['additional_cap_usd']
    with (ROOT / '.local/labelling.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remaining = ceiling - conservative_total()
        logger.info(f'# Downstream {stage}\n\n-- PI/OpenAI\n\nRemaining approved allowance ${remaining:.4f}.\nSHOULD: complete coverage; malformed replies remain failures, never scores.\nTODO validate: later evidence changes preferences more than repeated-answer variation.\n')
        if stage.startswith('actors'):
            packets = [p for p in read_rows(prepared / 'packets.jsonl') if (p['metadata']['original_source_id'] in SMOKE_IDS) == smoke]
            write_rows(out / 'packets.jsonl', packets)
            summary = asyncio.run(run_requests(packets, out, max_new_usd=min(3.5, remaining) if smoke else remaining,
                concurrency=12 if smoke else config['concurrency'], key_source=budget['key_source'], project_cap_usd=ceiling,
                evaluation_cap_usd=budget['additional_cap_usd'], reservation_usd=manifest['actor_reservation_usd'], scope=budget['scope']))
            parsed = [parse(r, config) for r in read_rows(out / 'receipts.jsonl')]
            write_rows(out / 'parsed.jsonl', parsed)
            summary['parse_status'] = dict(Counter(r['status'] for r in parsed))
            assert len(parsed) == len(packets)
        else:
            summary = {}
            for condition in CONDITIONS:
                actions = label_actions([r for r in scenes if r['condition'] == condition])
                if calibration:
                    actions = actions[:manifest['label_batch_size']]
                for kind in ('virtues', 'dnd'):
                    rubric, items, request_fn, parse_fn = judge_spec(kind)
                    if calibration:
                        items = items[:1]
                    destination = out / condition / kind
                    destination.mkdir(parents=True)
                    request = lambda r, i, acts, full_feedback: judge_request(request_fn, r, i, acts, full_feedback=full_feedback)
                    reservation = max(len(json.dumps(payload, ensure_ascii=False).encode()) * lv.TARIFF * 1.1
                        for _, _, _, payload in lv.packets(rubric, items, actions, manifest['label_batch_size'], request_fn=request))
                    remaining = ceiling - conservative_total()
                    assert remaining >= reservation
                    args = argparse.Namespace(out=destination, batch_size=manifest['label_batch_size'], concurrency=1 if calibration else 12,
                        key_source=budget['key_source'], budget_usd=ceiling, max_new_usd=remaining)
                    result = asyncio.run(lv.paid(args, rubric, items, actions, request_fn=request, parse_fn=parse_fn,
                        cache_name='downstream_' + kind + '_cache', reservation_usd=reservation, scope=budget['scope']))
                    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
                    summary[condition + '/' + kind] = result
                    logger.info(f'Completed {condition}/{kind}; new cost ${result["new_cost_usd"]:.5f}.')
        summary['allowance_used_conservative_usd'] = conservative_total() - budget['project_start_conservative_usd']
        assert summary['allowance_used_conservative_usd'] <= budget['additional_cap_usd']
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        logger.info(f'Completed {stage}. Inspect feedback before interpreting scores.\n{out / "run.md"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=['actors-smoke', 'labels-calibration-smoke', 'labels-smoke', 'actors-rest', 'labels-rest'], required=True)
    args = ap.parse_args()
    execute(args.prepared, args.out, args.stage)
