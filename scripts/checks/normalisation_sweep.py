"""Which per-question normalisation recovers consistent per-model preferences? Offline sweep on v12 (1,000 questions x 6 models x 2 answers).

Per answer, a statistic S[q, m, r, d] pairs a transform T of the stated percentages with within-question-centred option labels x~:
    cov:  S = sum_k T_k x~_k / K          corr: S = cov / (sd_k T * sd_k x~)
Profiles average S over questions. Separation is unit-free (all are 0 for indistinguishable models, see the shuffled null):
    rel_q  split-half reliability across disjoint question halves: per dim, corr over models of across-model-centred profiles; mean over dims
    rel_r  same, but answer 1 vs answer 2 on the same questions (test-retest; an upper bound on rel_q)
    id_q   nearest-neighbour model identification across question halves, chance 1/6
Selection: questions are hashed into explore/confirm halves; rank variants on explore rel_q, report confirm. Label-free part: are models separable
from their percentages alone (cross-answer corr of double-centred log-probabilities)? -- PI/OpenAI, 2026-10-29, from oracle advice in slop/reviews/20261029_norm_oracle_*.md
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/20261027_random1000_v12'
EPS = 0.005
N_SPLITS = 200
rng = np.random.default_rng(20261029)


def load():
    f = (RUN / 'question_tables.jsonl').open()
    dims = json.loads(f.readline())['_meta']['dimensions']
    qs = [json.loads(x) for x in f]
    models = sorted(qs[0]['models'])
    qs = [q for q in qs if all(len(q['models'].get(m, {}).get('draws', [])) == 2 for m in models)]
    K = max(len(q['options']) for q in qs)
    Q, M, D = len(qs), len(models), len(dims)
    P = np.zeros((Q, M, 2, K)); mask = np.zeros((Q, K), bool); X = np.zeros((Q, K, D))
    for a, q in enumerate(qs):
        keys = list(q['options'])
        mask[a, :len(keys)] = True
        X[a, :len(keys)] = np.array([q['option_scores'][k] for k in keys], float)
        for b, m in enumerate(models):
            for d in q['models'][m]['draws']:
                P[a, b, d['draw'], :len(keys)] = [d['probabilities'][k] / 100 for k in keys]
    assert np.allclose(P.sum(-1), 1, atol=.02), 'percentages must sum to 100'
    bad = (~np.isfinite(X) & mask[..., None]).any(1)  # D&D 'unclear' for some option: drop that (question, dim), do not impute
    print(f'dropped (question, dim) cells with an unlabelled option: {bad.sum()} of {bad.size}, all in', sorted({dims[i]["id"] for i in np.flatnonzero(bad.any(0))}))
    X = np.where(bad[:, None, :] | ~np.isfinite(X), 0., X)  # zero column -> no label spread -> stat is NaN for that (q, d)
    src = np.array([q['source'] for q in qs])
    explore = np.array([int(hashlib.sha256(q['source_id'].encode()).hexdigest()[0], 16) < 8 for q in qs])
    fav = np.array([-1. if d['kind'] == 'vices' else 1. for d in dims])
    return qs, dims, models, P, mask, X * fav, src, explore


def centre(A, mask, axis_k=-1):
    w = mask.astype(float)
    mu = (A * w).sum(axis_k, keepdims=True) / w.sum(axis_k, keepdims=True)
    return (A - mu) * w


def dim_sets(dims, X, mask, src):
    ids = [d['id'] for d in dims]
    direct = [i for i, d in enumerate(dims) if d['family'] == 'virtues' and d['derived_from'] is None]
    behav = [i for i, d in enumerate(dims) if d['family'] == 'behaviours']
    dnd = [ids.index('order'), ids.index('morality')]
    groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
    Xc = centre(X, mask[..., None], axis_k=1)
    G = np.stack([Xc[:, :, [ids.index(i) for i in members]].mean(-1) for members in groups.values()], -1)
    good = np.zeros(X.shape[:2] + (1,)); Gres = np.zeros_like(G)
    for s in np.unique(src):  # goodness = PC1 of option-centred direct virtues, per source; style = groups with goodness partialled out
        sel = src == s
        Z = Xc[sel][:, :, direct][mask[sel]]
        v = np.linalg.svd(Z, full_matrices=False)[2][0]
        v *= np.sign(v.sum())
        good[sel, :, 0] = (Xc[sel][:, :, direct] @ v) * mask[sel]
        g = good[sel][mask[sel]][:, 0]
        beta = (G[sel][mask[sel]] * g[:, None]).sum(0) / (g ** 2).sum()
        Gres[sel] = (G[sel] - good[sel] * beta) * mask[sel][..., None]
    return {'goodness': good, 'dnd': Xc[:, :, dnd], 'groups9': G, 'style9': Gres, 'virt66': Xc[:, :, direct], 'behav26': Xc[:, :, behav]}, list(groups)


def p_transforms(P, mask):
    m = mask[:, None, None, :]
    logp = np.log(np.where(m, P, 1) + EPS)
    lc = centre(logp, m)
    sd = np.sqrt((lc ** 2).sum(-1, keepdims=True) / m.sum(-1, keepdims=True))
    rank = np.where(m, (np.sign(P[..., :, None] - P[..., None, :]) * m[..., None, :]).sum(-1), 0)  # wins minus losses within menu
    top = np.where(m, (P == np.where(m, P, -1).max(-1, keepdims=True)).astype(float), 0)
    return {'p': centre(P, m), 'logp': lc, 'logp_z': np.divide(lc, sd, out=np.zeros_like(lc), where=sd > 0),
            'rank': centre(rank, m), 'top': centre(top / top.sum(-1, keepdims=True), m)}


def label_transforms(Xs, mask):
    m = mask[..., None]
    sd = np.sqrt((Xs ** 2).sum(1, keepdims=True) / m.sum(1, keepdims=True))
    rank = (np.sign(Xs[:, :, None, :] - Xs[:, None, :, :]) * m[:, None]).sum(2) * m
    return {'raw': Xs, 'z': np.divide(Xs, sd, out=np.zeros_like(Xs), where=sd > 1e-9), 'rank': centre(rank, m, axis_k=1)}


def stat(T, Xl, mask, kind):
    # T: (Q,M,R,K) centred; Xl: (Q,K,D) centred -> (Q,M,R,D); NaN where the question has no label spread
    n = mask.sum(-1)[:, None, None, None]
    cov = np.einsum('qmrk,qkd->qmrd', T, Xl) / n
    sx = np.sqrt((Xl ** 2).sum(1) / mask.sum(-1)[:, None])[:, None, None, :]
    if kind == 'cov':
        return np.where(sx > 1e-9, cov, np.nan)
    st = np.sqrt((T ** 2).sum(-1) / mask.sum(-1)[:, None, None])[..., None]
    return np.where((sx > 1e-9) & (st > 1e-9), cov / (st * sx + 1e-12), np.nan)


def profiles(S, W):
    # S: (Q,M,D) per-question stat; W: (n,Q) question weights -> (n,M,D), centred across models
    ok = np.isfinite(S)
    num = np.einsum('nq,qmd->nmd', W, np.where(ok, S, 0))
    den = np.einsum('nq,qmd->nmd', W, ok.astype(float))
    prof = num / np.where(den > 0, den, np.nan)
    return prof - np.nanmean(prof, 1, keepdims=True)


def corr_models(A, B):
    # per (split, dim) Pearson corr across models of centred profiles; mean over dims and splits
    num = (A * B).sum(1); den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    return np.nanmean(num / np.where(den > 0, den, np.nan))


def ident(A, B):
    sd = np.sqrt(np.nanmean(np.concatenate([A, B], 1) ** 2, 1, keepdims=True))
    A, B = A / sd, B / sd
    dist = ((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1)
    return (dist.argmin(-1) == np.arange(A.shape[1])).mean()


def separation(S, qw, sel):
    # S: (Q,M,R,D); qw: (Q,) weights; sel: questions in this cell
    idx = np.flatnonzero(sel)
    WA = np.zeros((N_SPLITS, len(sel))); WB = np.zeros_like(WA)
    for s in range(N_SPLITS):
        perm = rng.permutation(idx)
        WA[s, perm[:len(idx) // 2]] = qw[perm[:len(idx) // 2]]
        WB[s, perm[len(idx) // 2:]] = qw[perm[len(idx) // 2:]]
    Sm = np.nanmean(S, 2)
    A, B = profiles(Sm, WA), profiles(Sm, WB)
    W1 = (sel * qw)[None]
    R0, R1 = profiles(S[:, :, 0], W1), profiles(S[:, :, 1], W1)
    return {'rel_q': corr_models(A, B), 'id_q': ident(A, B), 'rel_r': corr_models(R0, R1)}


def main():
    qs, dims, models, P, mask, X, src, explore = load()
    sets, group_names = dim_sets(dims, X, mask, src)
    Ts = p_transforms(P, mask)
    pbar = P.mean((1, 2))
    weights = {'equal': np.ones(len(qs)), 'contested': 1 - pbar.max(-1)}
    out = []
    print(f'# Normalisation sweep\n\n{len(qs)} questions, models: {", ".join(m.split("/")[-1] for m in models)}; explore {explore.sum()} / confirm {(~explore).sum()}\n')

    # shuffled null: permute model identity independently per question (both answers move together)
    perm = np.stack([rng.permutation(len(models)) for _ in qs])
    Pnull = P[np.arange(len(qs))[:, None], perm]
    for source in ('machiavelli', 'daily_dilemmas'):
        for half, hsel in (('explore', explore), ('confirm', ~explore)):
            sel = (src == source) & hsel
            for setname, Xs in sets.items():
                for lt, Xl in label_transforms(Xs, mask).items():
                    for pt, T in Ts.items():
                        for kind in ('cov', 'corr'):
                            S = stat(T, Xl, mask, kind)
                            for wn, qw in weights.items():
                                out.append(dict(source=source, half=half, dims=setname, labels=lt, probs=pt, stat=kind, weight=wn, **separation(S, qw, sel)))
                # null for the plain baseline
                Sn = stat(p_transforms(Pnull, mask)['logp'], Xs, mask, 'cov')
                out.append(dict(source=source, half=half, dims=setname, labels='raw', probs='logp', stat='cov', weight='NULL-shuffled', **separation(Sn, weights['equal'], sel)))
    df = pl.DataFrame(out)
    df.write_parquet(RUN / 'normalisation_sweep.parquet')
    key = ['source', 'dims', 'labels', 'probs', 'stat', 'weight']
    wide = df.filter(pl.col('half') == 'explore').select(key + ['rel_q', 'id_q', 'rel_r']).join(
        df.filter(pl.col('half') == 'confirm').select(key + [pl.col('rel_q').alias('rel_q_confirm'), pl.col('id_q').alias('id_q_confirm')]), on=key)
    pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(20); pl.Config.set_float_precision(2); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_tbl_hide_column_data_types(True)
    for source in ('machiavelli', 'daily_dilemmas'):
        for setname in sets:
            w = wide.filter((pl.col('source') == source) & (pl.col('dims') == setname))
            base = w.filter((pl.col('labels') == 'raw') & (pl.col('probs') == 'logp') & (pl.col('stat') == 'cov') & pl.col('weight').is_in(['equal', 'NULL-shuffled']))
            top = w.filter(pl.col('weight') != 'NULL-shuffled').sort('rel_q', descending=True).head(5)
            print(f'\n## {source} / {setname}: baseline and null, then top 5 by explore rel_q (confirm columns are the honest numbers)\n')
            print(pl.concat([base, top]).drop('source', 'dims'))

    # label-free separability: double-centred log-probabilities, answer 1 vs answer 2 (each answer centred on that answer's consensus)
    print('\n## Label-free: are models separable from percentages alone?\n')
    m = mask[:, None, None, :]
    L = centre(np.log(np.where(m, P, 1) + EPS), m)
    dev = (L - L.mean(1, keepdims=True)) * m  # (Q,M,R,K) deviation from the same-answer consensus
    rows = []
    for source in ('machiavelli', 'daily_dilemmas'):
        for nopt in ('all', '2', '3+'):
            sel = (src == source) & (True if nopt == 'all' else (mask.sum(-1) == 2) if nopt == '2' else (mask.sum(-1) >= 3))
            mm = np.broadcast_to(mask[sel][:, None, :], dev[sel][:, :, 0].shape)
            a, b = dev[sel][:, :, 0][mm], dev[sel][:, :, 1][mm]
            per_model = []
            for j, mod in enumerate(models):
                aa, bb = dev[sel][:, j, 0][mask[sel]], dev[sel][:, j, 1][mask[sel]]
                per_model.append(np.corrcoef(aa, bb)[0, 1])
            rows.append(dict(source=source, options=nopt, questions=int(sel.sum()), cross_answer_corr=float(np.corrcoef(a, b)[0, 1]),
                             **{mod.split('/')[-1][:10]: float(r) for mod, r in zip(models, per_model)}))
    print(pl.DataFrame(rows))
    print('\nRead: corr of a model\'s deviation from consensus in answer 1 with its deviation in answer 2 (same question). 0 = deviations are pure noise.')


if __name__ == '__main__':
    main()
