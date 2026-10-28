"""Print, per model, the game/everyday questions where both of its answers depart furthest from the other models in the same way
(score = sum_k dev1_k * dev2_k on double-centred log p). Chosen extremes for reading, not a representative sample. -- PI/OpenAI, 2026-10-29"""

import sys

import numpy as np

from normalisation_sweep import EPS, centre, load

SOURCE = sys.argv[1] if len(sys.argv) > 1 else 'machiavelli'
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
qs, dims, models, P, mask, X, src, explore = load()
m4 = mask[:, None, None, :]
L = centre(np.log(np.where(m4, P, 1) + EPS), m4)
dev = (L - L.mean(1, keepdims=True)) * m4
score = (dev[:, :, 0] * dev[:, :, 1]).sum(-1)  # (Q,M)
good = [d['id'] for d in dims].index('morality')
for j, mod in enumerate(models):
    print(f'\n################ {mod}')
    for q in np.argsort(-np.where(src == SOURCE, score[:, j], -np.inf))[:TOP]:
        row = qs[q]; keys = list(row['options'])
        others = np.delete(P[q], j, 0).mean((0, 1))
        print(f'\n=== {row["source_id"]}  (score {score[q, j]:.2f})\n...{row["text"][-700:]}\n')
        for i, k in enumerate(keys):
            g = X[q, i, good]
            print(f'  [{k}] others {others[i]:4.0%} | this model {P[q, j, 0, i]:4.0%} {P[q, j, 1, i]:4.0%} | D&D good {g:+.1f} | {row["options"][k][:110]}')
        for d in row['models'][mod]['draws']:
            print(f'  reason{d["draw"]}: {d["reasoning"][:420]}')
