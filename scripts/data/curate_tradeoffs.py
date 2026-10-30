"""Apply explicit, exact-match edits to selected generated families. -- PI/OpenAI"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from dataset_io import ROOT, write_jsonl

sys.path.insert(0, str(ROOT / 'scripts/generation'))
from generate_tradeoffs import validate


def curate(selection: Path, inputs: Path, out: Path) -> None:
    spec = json.loads(selection.read_text())
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == spec['inputs_sha256']
    sources = {r['source_id']: r for r in map(json.loads, inputs.open())}
    families = []
    for item in spec['families']:
        receipts = [r for r in map(json.loads, (ROOT / item['receipt_path']).open()) if r['request_key'] == item['request_key']]
        assert len(receipts) == 1 and receipts[0]['metadata']['source_id'] == item['source_id']
        family = json.loads(receipts[0]['response']['choices'][0]['message']['content'])
        for patch in item['edits']:
            assert patch['reason']
            if not patch['path']:
                assert family == patch['old'], item['source_id']
                family = copy.deepcopy(patch['new'])
                continue
            container = family
            for key in patch['path'][:-1]:
                container = container[key]
            key = patch['path'][-1]
            assert container[key] == patch['old'], (item['source_id'], patch['path'])
            container[key] = copy.deepcopy(patch['new'])
        validate(family, sources[item['source_id']])
        families.append({'source_id': item['source_id'], 'request_key': item['request_key'], **family})
    assert len(families) == len({f['source_id'] for f in families})
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'families.jsonl', families)
    (out / 'curation.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    manifest = {'author': 'PI/OpenAI', 'families': len(families), 'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                'families_sha256': hashlib.sha256((out / 'families.jsonl').read_bytes()).hexdigest(),
                'status': 'curated; semantic approval still required'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=ROOT / 'data/interim/tradeoff12_sources_v2/generation_inputs.jsonl')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    curate(a.selection, a.inputs, a.out)
