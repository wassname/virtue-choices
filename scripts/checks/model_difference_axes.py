"""Which label directions separate the models most? PCA of across-model-centred profiles (cov of centred log p with option labels,
mean over explore questions), then check on confirm questions whether models keep their order along each axis. Labels are
z-scored per dimension across options first so units do not decide the axes. -- PI/OpenAI, 2026-10-29"""

import numpy as np

from normalisation_sweep import dim_sets, label_transforms, load, p_transforms, profiles, stat

qs, dims, models, P, mask, X, src, explore = load()
sets, groups = dim_sets(dims, X, mask, src)
names = {'virt66': [d['label'] for d in dims if d['family'] == 'virtues' and d['derived_from'] is None],
         'behav26': [d['id'] for d in dims if d['family'] == 'behaviours']}
import sys
PT = sys.argv[1] if len(sys.argv) > 1 else 'logp'  # logp_z removes each answer's sharpness (log p scaled to unit sd over options)
T = p_transforms(P, mask)[PT]
print('probability transform:', PT)
short = [m.split('/')[-1][:8] for m in models]
for source in ('machiavelli', 'daily_dilemmas'):
    Xs = np.concatenate([sets['virt66'], sets['behav26']], -1)
    lab = names['virt66'] + names['behav26']
    Xz = label_transforms(Xs, mask)['z']
    S = np.nanmean(stat(T, Xz, mask, 'cov'), 2)  # (Q,M,D)
    E = profiles(S, ((src == source) & explore)[None].astype(float))[0]
    C = profiles(S, ((src == source) & ~explore)[None].astype(float))[0]
    E, C = np.nan_to_num(E), np.nan_to_num(C)
    U, sv, Vt = np.linalg.svd(E, full_matrices=False)
    print(f'\n## {source}: between-model axes from explore questions (share of between-model variance), checked on confirm\n')
    for k in range(3):
        v = Vt[k]
        e, c = E @ v, C @ v
        r = np.corrcoef(e, c)[0, 1]
        top = np.argsort(-np.abs(v))[:8]
        print(f'axis {k + 1}: share {sv[k] ** 2 / (sv ** 2).sum():.0%}, explore-vs-confirm corr of model scores {r:+.2f}')
        print('   model scores (explore | confirm): ' + ', '.join(f'{n} {a:+.3f}|{b:+.3f}' for n, a, b in sorted(zip(short, e, c), key=lambda x: -x[1])))
        print('   top loadings: ' + ', '.join(f'{lab[i]} {v[i]:+.2f}' for i in top))
