"""Plot matched/fixed-label virtue profiles and paired model contrasts. -- PI/OpenAI"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = {'anthropic/claude-haiku-5.5': 'Haiku', 'deepseek/deepseek-v4.1-flash': 'DeepSeek',
         'google/gemini-3.8-flash': 'Gemini', 'meta/muse-spark-1.3-contributor': 'Muse',
         'openai/gpt-5.6-luna': 'Luna', 'qwen/qwen3.8-27b': 'Qwen'}
TITLES = ['No outcome text', 'Text to next decision', 'Plus later continuations']


def save(fig, path: Path) -> None:
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    assert box.x0 > -.02 and box.y0 > -.02 and box.x1 < w + .02 and box.y1 < h + .02, 'Clipped figure'
    fig.savefig(path.with_suffix('.png'), dpi=170)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def plot(out: Path) -> None:
    result = json.loads((out / 'estimates.json').read_text())
    models, groups = result['models'], result['display_groups']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#333333']
    ix = [result['dimensions'].index(g) for g in groups]
    profiles = {(r['model'], r['choice_condition'], r['label_condition']):np.array(r['means'])[ix]
                for r in result['profiles'] if r['mode']=='stated'}
    all_values = np.concatenate(list(profiles.values()))
    low = min(0, np.floor(all_values.min()*10)/10)
    high = max(.1, np.ceil(all_values.max()*10)/10)
    angles = np.linspace(0, 2*np.pi, len(groups), endpoint=False)
    ring = np.r_[angles, angles[0]]
    for scoring in ('matched','fixed-none'):
        fig, axes = plt.subplots(1,3,subplot_kw={'projection':'polar'},figsize=(16,6),layout='constrained')
        for ci,(ax,condition) in enumerate(zip(axes,result['conditions'],strict=True)):
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles,groups,fontsize=9)
            ax.tick_params(axis='x',pad=10)
            ax.set_ylim(0,high-low)
            ticks=np.linspace(low,high,5)
            ax.set_yticks(ticks-low,[f'{v:.2f}' for v in ticks],fontsize=8,color='#666666')
            ax.set_rlabel_position(12)
            ax.grid(color='#dddddd',linewidth=.7)
            ax.spines['polar'].set_visible(False)
            for mi,model in enumerate(models):
                value=profiles[model,condition,condition if scoring=='matched' else 'none']
                ax.plot(ring,np.r_[value,value[0]]-low,color=colors[mi],lw=1.6,label=NAMES[model])
            ax.set_title(TITLES[ci],pad=25,fontsize=12)
        label_note='Labels match the evidence shown' if scoring=='matched' else 'All choices scored with no-outcome labels'
        fig.suptitle(f'Virtue preferences in {len(result["families"])} selected scenes\n{label_note}; score above uniform choice',fontsize=14)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=6,frameon=False)
        save(fig,out/f'virtue_profiles_{scoring}')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    for ax,scoring,label in zip(axes,('matched','fixed-none'),('Evidence-matched labels','Fixed no-outcome labels'),strict=True):
        r=next(r for r in result['separation'] if r['mode']=='stated' and r['scoring']==scoring)
        y=np.array(r['rms_model_gap']);ci=np.array(r['rms_gap_cluster_ci95']);x=np.arange(3)
        ax.vlines(x,ci[0],ci[1],color='#0072B2',lw=1.4)
        ax.plot(x,y,'o-',color='#0072B2',label='Observed gap; descriptive game-bootstrap 95% interval')
        null=np.array(r['permutation_null_rms_ci95'])
        ax.fill_between(x,null[0],null[1],color='#aaaaaa',alpha=.3,label='Shuffled model identities: central 95%')
        ax.plot(x,r['permutation_null_rms_median'],color='#777777',lw=1)
        ax.plot(x,r['repeat_noise_rms_for_gap_of_means'],'x:',color='#D55E00',label='Repeat-noise scale for a gap of means')
        ax.set_xticks(range(3),['None','Next decision','Plus later'])
        ax.set_ylim(bottom=0)
        ax.set_title(label)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Between-model profile gap (root mean square)')
    fig.suptitle('Model differences against shuffled identities and repeat variability',fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',frameon=False,fontsize=9)
    save(fig,out/'model_separation')
    pairs=[r for r in result['pairs'] if r['mode']=='stated' and r['scoring']=='fixed-none']
    fig,axes=plt.subplots(3,3,figsize=(15,12),sharex=True,sharey=True,layout='constrained')
    y=np.arange(len(pairs))[::-1]
    pair_labels=[NAMES[r['a']]+' − '+NAMES[r['b']] for r in pairs]
    for gi,(group,ax) in enumerate(zip(groups,axes.flat,strict=True)):
        point=np.array([r['later_minus_none'][gi] for r in pairs])
        ci=np.array([np.array(r['later_minus_none_cluster_ci95'])[:,gi] for r in pairs])
        ax.axvline(0,color='#999999',lw=.8)
        ax.hlines(y,ci[:,0],ci[:,1],color='#bbbbbb',lw=1)
        ax.scatter(point,y,color='#0072B2',s=16)
        ax.set_yticks(y,pair_labels,fontsize=8)
        ax.set_title(group)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    fig.suptitle('How later text changes each model-pair difference\nFixed no-outcome labels; pointwise 95% game-cluster intervals, not simultaneous',fontsize=14)
    fig.supxlabel('(Model A − model B) with later text, minus the same difference without outcomes')
    save(fig,out/'paired_condition_changes')
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    plot(parser.parse_args().out)
