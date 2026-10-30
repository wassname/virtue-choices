"""Run trade-off stages under one approved additional allowance. -- PI/OpenAI"""

import asyncio
import fcntl
import json
import sys
from pathlib import Path

from loguru import logger

from eval_api import ROOT, run_requests
from billing_reservations import outstanding

BUDGET = ROOT / '.local/tradeoff_pilot_budget.json'


def run_batch(packets: list[dict], out: Path, max_usd: float, reservation: float, concurrency: int = 8) -> dict:
    budget = json.loads(BUDGET.read_text())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
    (out / 'packets.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in packets))
    logger.remove()
    logger.add(sys.stdout, format='{message}', level='INFO')
    logger.add(out / 'run.md', format='{message}', level='DEBUG')
    with (ROOT / '.local/labelling.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = sum(json.loads(x)['cost_usd'] for x in (ROOT / '.local/billing.jsonl').open()) + outstanding(ROOT)
        ceiling = budget['project_start_usd'] + budget['additional_cap_usd']
        remaining = min(max_usd, ceiling - current)
        assert remaining >= reservation, (remaining, reservation)
        logger.info(f'# Trade-off pilot stage\n\n-- PI/OpenAI\n\nSHOULD: {len(packets)} complete receipts. Stage cap ${remaining:.4f}; request reservation ${reservation:.4f}.')
        summary = asyncio.run(run_requests(packets, out, max_new_usd=remaining, concurrency=concurrency,
            key_source=budget['key_source'], project_cap_usd=ceiling,
            evaluation_cap_usd=budget['additional_cap_usd'], reservation_usd=reservation, scope=budget['scope']))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logger.info(f"Completed {len(packets)} requests; new cost ${summary['new_cost_usd']:.5f}.\n{out / 'run.md'}")
    return summary
