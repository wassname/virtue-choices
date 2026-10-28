// What each model's written reasoning appeals to (Jev judge, 8 kinds; 6 shown), mean share with 95% interval over questions.
// Data: outputs/20261029_reason_types_v12/reason_types_site.json via `just site-build`. Hover a model to highlight it. -- PI/OpenAI
import { useState } from 'react'
import { COLORS, FULL, meta } from './AlignmentChart'

type Row = { mean: number[]; lo: number[]; hi: number[]; questions: number }
export type ReasonData = { kinds: string[]; labels: string[]; sources: Record<'games' | 'everyday', Record<string, Row>> }

const INK = '#2f2e2a'
const pct = (v: number) => `${Math.round(100 * v)}%`

export default function ReasonTypes({ data }: { data: ReasonData }) {
  const [hover, setHover] = useState<string | null>(null)
  const models = Object.keys(data.sources.games)
  const L = 250, PW = 330, GAP = 40, rowH = 58, top = 34, xmax = .6
  const H = top + rowH * data.kinds.length + 40
  const x = (p: number, v: number) => L + p * (PW + GAP) + v / xmax * PW
  return (
    <div>
      <svg viewBox={`0 0 ${L + 2 * PW + GAP + 10} ${H}`} width="100%" role="img" aria-label="What models say they weigh">
        {(['games', 'everyday'] as const).map((src, p) => <g key={src}>
          <text x={x(p, 0)} y={16} fontSize={15} fill={INK}>{src === 'games' ? 'Game scenes' : 'Everyday dilemmas'}</text>
          {[0, .1, .2, .3, .4, .5, .6].map(t => <g key={t}>
            <line x1={x(p, t)} x2={x(p, t)} y1={top - 6} y2={H - 34} stroke="#e6e3da" />
            <text x={x(p, t)} y={H - 18} fontSize={11} fill="#77746a" textAnchor="middle">{pct(t)}</text></g>)}
          {data.kinds.map((_, k) => models.map((m, j) => {
            const r = data.sources[src][m]
            const y = top + k * rowH + 8 + j * 7
            const dim = hover && hover !== m
            return <g key={m} opacity={dim ? .15 : 1} onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)}>
              <line x1={x(p, r.lo[k])} x2={x(p, r.hi[k])} y1={y} y2={y} stroke={COLORS[m]} strokeWidth={1.5} />
              <circle cx={x(p, r.mean[k])} cy={y} r={4} fill={COLORS[m]}>
                <title>{`${FULL[m]}\n${data.labels[k]}: ${pct(r.mean[k])} (95% ${pct(r.lo[k])}–${pct(r.hi[k])}), ${r.questions} questions\n${meta(m)}`}</title></circle></g>
          }))}
          <text x={x(p, .3)} y={H - 2} fontSize={11} fill="#77746a" textAnchor="middle">share of the written reasoning that appeals to this</text>
          <text x={x(p, .3)} y={H - 40} fontSize={10} fill="#d8d5cc" fontStyle="italic" textAnchor="middle">wassname/virtue-choices</text>
        </g>)}
        {data.labels.map((l, k) => { const [a, b] = l.split(' ('); return <g key={l}>
          <text x={L - 12} y={top + k * rowH + 22} fontSize={13} fill={INK} textAnchor="end">{a}</text>
          <text x={L - 12} y={top + k * rowH + 38} fontSize={11} fill="#77746a" textAnchor="end">({b}</text></g> })}
      </svg>
      <div className="legend">{models.map(m => <span key={m} onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)} title={`${FULL[m]}\n${meta(m)}`}
        style={{ color: COLORS[m], marginRight: '1em', cursor: 'default', opacity: hover && hover !== m ? .3 : 1 }}>● {m}</span>)}</div>
    </div>
  )
}
