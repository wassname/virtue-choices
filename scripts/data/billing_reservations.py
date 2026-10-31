"""Keep unknown fees and native usage estimates separate from charge receipts. -- PI/Sol and PI/OpenAI"""

import hashlib
import json
import math
from pathlib import Path

PROJECT_CAP_USD = 80
EVALUATION_CAP_USD = 12.25
EVALUATION_SCOPE = "action-choice-eval-v1"
NATIVE_USAGE_SCOPES = {EVALUATION_SCOPE, "virtue-labels-v2", "legacy-unassigned"}


def reserve_unknown_fee(event: dict, out: Path, reservation_usd: float = .01,
                        scope: str = EVALUATION_SCOPE, *, root: Path = Path(__file__).resolve().parents[2]) -> float:
    path = root / '.local/billing_reservations.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    identity = json.dumps({'payload_key': event['request_key'], 'time': event['time']},
                          sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    reservation = {'request_key': hashlib.sha256(identity.encode()).hexdigest(),
                   'payload_request_key': event['request_key'], 'scope': scope,
                   'reserved_usd': reservation_usd, 'run': str(out), 'attempt': event['attempt'],
                   'reason': 'Request may have been billed; no charge receipt returned'}
    headers = event.get('response_headers', {})
    if 'x-generation-id' in headers:
        reservation['response_id'] = headers['x-generation-id']
    rows.append(reservation)
    path.write_text(json.dumps(rows, indent=2) + '\n')
    return reservation_usd


def unknown_fees(root: Path, scope: str | None = None) -> float:
    path = root / ".local/billing_reservations.json"
    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text())
    identities = [("response", row["response_id"]) if "response_id" in row else ("request", row["request_key"]) for row in rows]
    assert len(set(identities)) == len(rows)
    assert all(math.isfinite(row["reserved_usd"]) and row["reserved_usd"] >= 0 for row in rows)
    return sum(row["reserved_usd"] for row in rows if scope is None or row["scope"] == scope)


def noncash_reviews(root: Path, scope: str | None = None) -> float:
    path = root / ".local/review_usage_estimates.json"
    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text())
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert all(row["scope"] in NATIVE_USAGE_SCOPES for row in rows), "Unknown native usage scope"
    values = [row["reported_cost_usd"]["costUsd"] for row in rows]
    assert all(math.isfinite(value) and value >= 0 for value in values)
    return sum(row["reported_cost_usd"]["costUsd"] for row in rows
               if scope is None or row["scope"] == scope)


def outstanding(root: Path, scope: str | None = None) -> float:
    return unknown_fees(root, scope) + noncash_reviews(root, scope)
