"""Is the sameness caused by the prompt? Label-free diversity between models for each prompt version, on common questions.
Per question, single answers only (so between and within use the same noise):
    between = mean over model pairs and answer pairs of D(p_a, p_b);  within = mean over models of D(answer 1, answer 2)
    excess = between - within (0 = models differ no more than a model differs from itself)
D = Jensen-Shannon divergence (nats, percentages smoothed by +0.5%) and top-choice disagreement. 95% intervals: bootstrap over questions.
-- PI/OpenAI, 2026-10-29"""

import glob
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RUNS = {'v6 isekai': ['outputs/20261026_random1000_isekai'], 'v10': ['outputs/20261027_random1000_v10'],
        'v12': ['outputs/20261027_random1000_v12', 'outputs/20261027_random1000_v12_haiku', 'outputs/20261028_random1000_v12_grok'],
        'two-line (v13 short)': ['outputs/20261028_short_prompt_pilot/games']}
rng = np.random.default_rng(20261029)


def load(dirs, protocol=None):
    out = defaultdict(dict)  # (source_id, model) -> {draw: probs}
    for d in dirs:
        for f in glob.glob(str(ROOT / d / '**/choices.jsonl'), recursive=True):
            if 'failed' in f:
                continue
            for r in map(json.loads, open(f)):
                if r['status'] == 'ok' and (protocol is None or r['actor_prompt_protocol'] == protocol):
                    out[r['source_id'], r['model']][r['draw']] = r['original_probabilities']
    return out


def js(a, b):
    p = np.array([a[k] for k in sorted(a)], float) + .5; q = np.array([b[k] for k in sorted(a)], float) + .5
    p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return .5 * (p * np.log(p / m)).sum() + .5 * (q * np.log(q / m)).sum()


def top(a, b):
    return float(max(a, key=a.get) != max(b, key=b.get))


def per_question(data, qs, models):
    rows = []
    for q in qs:
        for name, D in (('JS', js), ('top differs', top)):
            btw = [D(data[q, a][i], data[q, b][j]) for a, b in itertools.combinations(models, 2) for i in (0, 1) for j in (0, 1)]
            wit = [D(data[q, m][0], data[q, m][1]) for m in models]
            rows.append((name, np.mean(btw), np.mean(wit)))
    return rows


def summarise(label, data, qs, models):
    res = []
    rows = per_question(data, qs, models)
    for name in ('JS', 'top differs'):
        b = np.array([r[1] for r in rows if r[0] == name]); w = np.array([r[2] for r in rows if r[0] == name])
        boot = [(lambda i: (b[i] - w[i]).mean())(rng.integers(0, len(b), len(b))) for _ in range(2000)]
        res.append(dict(prompt=label, metric=name, questions=len(b), models=len(models), between=b.mean(), within=w.mean(),
                        excess=(b - w).mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), ratio=b.mean() / w.mean()))
    return res


def complete(data, models):
    qs = {q for q, _ in data}
    return {q for q in qs if all(len(data.get((q, m), {})) == 2 for m in models)}


def main():
    data = {k: load(v, 'isekai-v13-short' if 'short' in k else None) for k, v in RUNS.items()}
    five = ['deepseek/deepseek-v4.1-flash', 'meta/muse-spark-1.3-contributor', 'qwen/qwen3.8-27b', 'google/gemini-3.8-flash', 'openai/gpt-5.6-luna']
    six = five + ['anthropic/claude-haiku-5.5']
    rows = []
    for source in ('machiavelli', 'daily_dilemmas'):
        common = set.intersection(*(complete(data[k], five) for k in ('v6 isekai', 'v10', 'v12')))
        common = sorted(q for q in common if q.startswith(source))
        for k in ('v6 isekai', 'v10', 'v12'):
            rows += [r | {'source': source, 'set': 'same 5 models, common questions'} for r in summarise(k, data[k], common, five)]
    short_q = sorted(complete(data['two-line (v13 short)'], six) & complete(data['v12'], six))
    for k in ('v12', 'two-line (v13 short)'):
        rows += [r | {'source': 'machiavelli', 'set': f'6 models, {len(short_q)} pilot game questions'} for r in summarise(k, data[k], short_q, six)]
    # who is the outlier under v12: each model's mean single-answer JS to the others, minus the others' mean within-model JS
    seven = six + ['x-ai/grok-4.3']
    for source in ('machiavelli', 'daily_dilemmas'):
        qs = sorted(q for q in complete(data['v12'], seven) if q.startswith(source))
        for m in seven:
            d = [np.mean([js(data['v12'][q, m][i], data['v12'][q, o][j]) for o in seven if o != m for i in (0, 1) for j in (0, 1)]) - np.mean([js(data['v12'][q, o][0], data['v12'][q, o][1]) for o in seven]) for q in qs]
            rows.append(dict(prompt='v12', metric='JS excess, this model vs the others', source=source, set=m.split('/')[-1], questions=len(qs), models=7, excess=float(np.mean(d))))
    pl.Config.set_tbl_formatting('MARKDOWN'); pl.Config.set_float_precision(3); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(14); pl.Config.set_fmt_str_lengths(40)
    df = pl.DataFrame(rows)
    print('# Between-model diversity by prompt version (label-free; single answers)\n')
    print(df.filter(pl.col('metric') != 'JS excess, this model vs the others').select('source', 'set', 'prompt', 'metric', 'questions', 'between', 'within', 'excess', 'lo', 'hi', 'ratio'))
    print('\n## v12: which model differs most from the rest (JS excess over within-model noise)\n')
    print(df.filter(pl.col('metric') == 'JS excess, this model vs the others').select('source', 'set', 'questions', 'excess').sort('source', 'excess', descending=[False, True]))


if __name__ == '__main__':
    main()
