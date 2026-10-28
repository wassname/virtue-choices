"""How much of each model's consistent deviation from the other models do the option labels explain? Offline, v12.

Target: dev[q,m,r,k] = log p centred over options, minus the same-answer mean over models (double-centred; consensus as reference, not DeepSeek).
Predictors, fit per model on explore questions, scored on confirm questions (fixed hash split, R^2 of the 2-answer mean):
    sharp:   dev ~ (tau_m - 1) * consensus            (the model is just sharper / flatter than the crowd; no labels)
    labels:  dev ~ x~ . beta_m   (ridge)              (conditional-logit taste with shared option intercept a_qk = consensus)
    both
Noise ceiling: reliable share of var(mean of 2 answers) = cov(answer1, answer2) / var(mean). -- PI/OpenAI, 2026-10-29
"""

from pathlib import Path

import numpy as np
import polars as pl

from normalisation_sweep import EPS, centre, dim_sets, load

LAMBDAS = [0.1, 1, 10, 100, 1000, 1e4, 1e5]
EMB = Path(__file__).resolve().parents[2] / 'data/interim/option_embeddings_v1.npz'


SETS = ('goodness', 'dnd', 'style9', 'groups9', 'virt66', 'behav26', 'emb_alone', 'emb_scene')


def embedding_sets(qs, mask, k=128):
    # option text embeddings, centred within question, top-k PCA components scaled to unit sd (unsupervised: no model answers used)
    z = np.load(EMB)
    row = {s: i for i, s in enumerate(z['ids'])}
    out = {}
    for name in ('alone', 'scene'):
        E = np.zeros((len(qs), mask.shape[1], z[name].shape[1]))
        for a, q in enumerate(qs):
            for b, key in enumerate(q['options']):
                E[a, b] = z[name][row[f"{q['source_id']}|{key}"]]
        E = (E - (E * mask[..., None]).sum(1, keepdims=True) / mask.sum(1)[:, None, None]) * mask[..., None]
        V = np.linalg.svd(E[mask], full_matrices=False)[2][:k]
        F = E @ V.T
        out['emb_' + name] = F / F[mask].std(0) * mask[..., None]
    return out


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / (y ** 2).sum()


def main():
    qs, dims, models, P, mask, X, src, explore = load()
    sets, _ = dim_sets(dims, X, mask, src)
    sets |= embedding_sets(qs, mask)
    m4 = mask[:, None, None, :]
    summary = []
    for target, f in (('logp eps=.005', lambda P: np.log(np.where(m4, P, 1) + EPS)), ('logp eps=.05', lambda P: np.log(np.where(m4, P, 1) + .05)),
                      ('p', lambda P: P)):
        df = fit(models, P, mask, src, explore, sets, centre(f(P), m4) * m4)
        if target.endswith('.005'):
            report(df)
        best = pl.max_horizontal(*[pl.col(c) for c in df.columns if c.endswith('+sharp')])  # includes embeddings
        summary.append(df.select('source', 'model', 'ceiling', 'sharp', best.alias('best_any+sharp'), 'virt66', 'emb_scene').with_columns(target=pl.lit(target)))
    print('\n## Robustness: target scale (log with small/large smoothing, plain probability)\n')
    print(pl.concat(summary).sort('source', 'model', 'target'))


def fit(models, P, mask, src, explore, sets, L):
    m4 = mask[:, None, None, :]
    cons = L.mean(1, keepdims=True)
    dev = (L - cons) * m4                      # (Q,M,R,K)
    dbar, cbar = dev.mean(2), cons.mean(2)[:, 0]  # (Q,M,K), (Q,K)
    rows = []
    for source in ('machiavelli', 'daily_dilemmas'):
        tr, te = (src == source) & explore, (src == source) & ~explore
        for j, mod in enumerate(models):
            name = mod.split('/')[-1][:8]
            mtr, mte = mask[tr], mask[te]
            y_tr, y_te = dbar[tr, j][mtr], dbar[te, j][mte]
            a, b = dev[te, j, 0][mte], dev[te, j, 1][mte]
            ceiling = np.mean(a * b) / np.mean(y_te ** 2)  # dev is mean-zero within question
            c_tr, c_te = cbar[tr][mtr], cbar[te][mte]
            k = (c_tr @ y_tr) / (c_tr @ c_tr)
            row = dict(source=source, model=name, n_test_q=int(te.sum()), ceiling=ceiling, tau=1 + k, sharp=r2(y_te, k * c_te))
            for setname in SETS:
                Xtr, Xte = sets[setname][tr][mtr], sets[setname][te][mte]
                best = max(LAMBDAS, key=lambda lam: inner_cv(Xtr, y_tr, lam, tr, mtr))
                beta = np.linalg.solve(Xtr.T @ Xtr + best * np.eye(Xtr.shape[1]), Xtr.T @ y_tr)
                row[setname] = r2(y_te, Xte @ beta)
                Z_tr, Z_te = np.column_stack([Xtr, c_tr]), np.column_stack([Xte, c_te])
                bz = np.linalg.solve(Z_tr.T @ Z_tr + best * np.diag([1.] * Xtr.shape[1] + [0.]), Z_tr.T @ y_tr)
                row[setname + '+sharp'] = r2(y_te, Z_te @ bz)
            rows.append(row)
    return pl.DataFrame(rows)


def report(df):
    pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(14); pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(30); pl.Config.set_float_precision(3)
    pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_tbl_hide_column_data_types(True)
    print('# Held-out R^2 of each model\'s deviation from consensus (confirm questions)\n')
    print('ceiling = most any predictor could explain (2-answer noise); sharp = label-free sharpness; other columns = labels only, ridge chosen by inner CV on explore\n')
    print(df.select('source', 'model', 'n_test_q', 'ceiling', 'tau', 'sharp', *SETS))
    print('\n## with sharpness added\n')
    print(df.select('source', 'model', 'sharp', *[c for c in df.columns if c.endswith('+sharp')]))


def inner_cv(Xtr, y_tr, lam, tr, mtr):
    # 2-fold by question inside the explore half; rows are grouped by question via mtr
    qid = np.repeat(np.arange(mtr.shape[0]), mtr.sum(1))
    fold = qid % 2 == 0
    score = 0
    for f in (fold, ~fold):
        beta = np.linalg.solve(Xtr[f].T @ Xtr[f] + lam * np.eye(Xtr.shape[1]), Xtr[f].T @ y_tr[f])
        score += r2(y_tr[~f], Xtr[~f] @ beta)
    return score


if __name__ == '__main__':
    main()
