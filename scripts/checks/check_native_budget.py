"""Unpaid native-API billing regressions using simulated HTTP responses. -- PI/OpenAI"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
import label_virtues as lv
from billing_reservations import outstanding


def check(case: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / '.local').mkdir()
        ledger = root / '.local/billing.jsonl'
        ledger.write_text('')
        out = root / 'run'
        out.mkdir()
        calls = []
        def respond(request):
            calls.append(request)
            if case == 'read-timeout':
                raise httpx.ReadTimeout('Simulated ambiguous timeout', request=request)
            if case == 'retry-cap':
                return httpx.Response(504, json={'error': 'timeout'})
            if case == 'missing-charge':
                return httpx.Response(200, json={'id': 'fixture'})
            first = len(calls) == 1
            return httpx.Response(500 if first else 200, json={'id': str(len(calls)), 'model': 'fixture',
                'provider': 'TypeSafe', 'usage': {'cost': .004 if first else .003},
                'answers': {'debug_word': {'choice': 'none'}}})
        client = httpx.AsyncClient
        args = argparse.Namespace(out=out, concurrency=1, batch_size=1, key_source='env',
            budget_usd=.1, max_new_usd=.015 if case == 'retry-cap' else .1)
        error = None
        with patch.object(lv, 'ROOT', root), patch.object(lv, 'load_dotenv'), \
             patch.dict(os.environ, {'OPENROUTER_API_KEY': 'offline-fixture'}), \
             patch.object(lv.httpx, 'AsyncClient', side_effect=lambda **kw: client(transport=httpx.MockTransport(respond), **kw)), \
             patch.object(lv.asyncio, 'sleep', new_callable=AsyncMock):
            try:
                result = asyncio.run(lv.paid(args, {'contraries': [], 'served_model': 'fixture', 'version': 'fixture'},
                    [{'item_id': 'virtue'}], [{'source_id': 'scene', 'choice_id': '0'}],
                    request_fn=lambda *a, **kw: {'model': 'fixture'},
                    parse_fn=lambda *a: [{'source_id': 'scene', 'choice_id': '0'}], scope='fixture'))
            except (AssertionError, httpx.ReadTimeout) as exc:
                error = exc
        if case == 'charged-retry':
            assert error is None and len(calls) == 2
            assert abs(result['new_cost_usd'] - .007) < 1e-12
            assert len(ledger.read_text().splitlines()) == 2
            assert outstanding(root) == 0
        else:
            assert error is not None and len(calls) == 1
            assert ledger.read_text() == '' and (out / 'labels.jsonl').read_text() == ''
            assert outstanding(root) == .01
            if case == 'retry-cap':
                assert 'Run cap reached during retries' in str(error)
        print('PASS', case)


if __name__ == '__main__':
    for case in ('retry-cap', 'read-timeout', 'missing-charge', 'charged-retry'):
        check(case)
