"""Text embeddings of every option (alone, and with the end of its scene) for a label-free test of whether models' consistent
deviations follow any learnable direction in option text. OpenRouter embeddings API; cached in data/interim. -- PI/OpenAI, 2026-10-29"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'openai/text-embedding-3-small'
OUT = ROOT / 'data/interim/option_embeddings_v1.npz'
TABLES = ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl'

qs = [json.loads(x) for x in list(TABLES.open())[1:]]
ids, alone, scened = [], [], []
for q in qs:
    scene = q['text'].split('Actions to choose:')[0][-1200:]
    for k, t in q['options'].items():
        ids.append(f"{q['source_id']}|{k}")
        alone.append(t)
        scened.append(f'Scene: {scene}\nChosen action: {t}')
key = json.loads((Path.home() / '.pi/agent/auth.json').read_text())['openrouter']['key']
vecs, cost = {}, 0.
with httpx.Client(headers={'Authorization': 'Bearer ' + key}, timeout=120) as c:
    for name, texts in (('alone', alone), ('scene', scened)):
        out = []
        for i in range(0, len(texts), 200):
            r = c.post('https://openrouter.ai/api/v1/embeddings', json={'model': MODEL, 'input': texts[i:i + 200]})
            r.raise_for_status()
            body = r.json()
            assert len(body['data']) == len(texts[i:i + 200])
            out += [d['embedding'] for d in sorted(body['data'], key=lambda d: d['index'])]
            cost += body['usage']['cost']
            with (ROOT / '.local/billing.jsonl').open('a') as f:
                f.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'request_key': f'embed-{name}-{i}', 'response_id': body.get('id', ''), 'cost_usd': body['usage']['cost'], 'model': MODEL, 'run': str(OUT)}) + '\n')
        vecs[name] = np.array(out, np.float32)
np.savez_compressed(OUT, ids=np.array(ids), **vecs)
print(f'{len(ids)} options x {vecs["alone"].shape[1]} dims; cost ${cost:.4f}; {OUT}')
