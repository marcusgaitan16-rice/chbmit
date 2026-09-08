"""
Figures for the CHB-MIT low-latency seizure decoder write-up.

Run in Colab after the calibration cache exists:

    exec(open('make_figures.py').read())
    fig1_calibration(results_iv)      # or evaluate(cache)
    fig2_loso(per_fold)               # from loso_folds.csv
    fig3_threshold(cache)             # needs the score cache
    fig4_corpus(m)                    # from the 8.2 verification table

Every figure is saved to /content/data/figures/ and Drive at 300 dpi.

DESIGN NOTES
------------
Colour carries meaning, not decoration: hard subjects are warm, controls are
cool grey. That single choice does most of the work in figure 1, because the
finding *is* the separation between those two groups.

Sample size is annotated on the figure itself, not left to the caption. chb09
rests on 3 test events and chb12 on 19; a reader should see that without
hunting for it.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = '/content/data/figures'
DRIVE_OUT = '/content/drive/MyDrive/chbmit/figures'

# hard subjects warm, controls cool grey
HARD = {'chb15': '#c0392b', 'chb12': '#d4691a', 'chb13': '#8e44ad', 'chb06': '#c2185b'}
CTRL = {'chb09': '#7f8c8d', 'chb10': '#95a5a6'}
COLORS = {**HARD, **CTRL}

mpl.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.8, 'lines.markersize': 5,
})


def _save(fig, name):
    for d in (OUT, DRIVE_OUT):
        try:
            os.makedirs(d, exist_ok=True)
            fig.savefig(f'{d}/{name}.png')
            fig.savefig(f'{d}/{name}.pdf')       # vector, for the write-up
        except Exception as e:
            print(f'  could not write {d}: {e}')
    print(f'saved {name}')


def _style(subj):
    """Hard subjects solid + circles; controls dashed + open squares."""
    if subj in CTRL:
        return dict(color=COLORS[subj], ls='--', marker='s', mfc='white', alpha=0.85)
    return dict(color=COLORS.get(subj, '#333'), ls='-', marker='o')


# =============================================================================
# Figure 1 — the calibration curve (headline)
# =============================================================================

def fig1_calibration(df, fname='fig1_calibration_curve', flag=(('chb06', 3),)):
    """df needs: subject, n, sens, lat, lift  (n_test_events optional).

    Three panels because sensitivity alone is not the story — a detector that
    catches everything 60 s late is not the same product as one that catches
    everything in 15 s.

    `flag` marks points that are known artifacts rather than results. chb06 at
    n=3 calibrates on a single leftover record and picks a bad quantile; leaving
    it unmarked would read as a real collapse when its AUPRC lift is in fact the
    highest of its series.
    """
    df = df.copy()
    if 'n_seizures' in df: df = df.rename(columns={'n_seizures': 'n'})
    if 'latency_s' in df:  df = df.rename(columns={'latency_s': 'lat'})
    flag = set(flag)

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    order = [s for s in list(HARD) + list(CTRL) if s in set(df.subject)]

    for subj in order:
        g = df[df.subject == subj].sort_values('n')
        st = _style(subj)
        nev = g.n_test_events.iloc[0] if 'n_test_events' in g else None
        lab = f'{subj} ({nev} ev)' if nev is not None else subj
        # single-point series (e.g. chb09) need a visible marker, not a line
        if len(g) == 1:
            st = {**st, 'ms': 9, 'mew': 1.6}
        ax[0].plot(g.n, g.sens, label=lab, **st)
        ax[1].plot(g.n, g.lat, **st)
        ax[2].plot(g.n, g.lift, **st)

    # mark known artifacts
    for subj, n in flag:
        r = df[(df.subject == subj) & (df.n == n)]
        if not len(r): continue
        ax[0].scatter(r.n, r.sens, s=150, facecolors='none',
                      edgecolors='#333', lw=1.3, zorder=5)
        ax[0].annotate('calibration\nartifact', (r.n.iloc[0], r.sens.iloc[0]),
                       textcoords='offset points', xytext=(-6, 16),
                       fontsize=7, ha='right', color='#333',
                       arrowprops=dict(arrowstyle='-', lw=0.7, color='#333'))

    ax[0].set_ylabel('event sensitivity')
    ax[0].set_ylim(-0.08, 1.15)
    ax[0].axhline(1.0, color='#bbb', lw=0.8, zorder=0)

    ax[1].set_ylabel('detection latency (s)')
    ax[1].axhline(4.0, color='#c0392b', lw=0.9, ls=':', zorder=0)
    ax[1].text(0.98, 0.04, 'architectural floor 4 s', transform=ax[1].transAxes,
               fontsize=7.5, color='#c0392b', ha='right')

    ax[2].set_ylabel('AUPRC lift over baseline')
    ax[2].set_yscale('log')
    ax[2].axhline(1.0, color='#666', lw=0.9, ls=':', zorder=0)
    ax[2].text(0.98, 0.04, 'chance', transform=ax[2].transAxes,
               fontsize=7.5, color='#666', ha='right')

    for a in ax:
        a.set_xlabel('patient seizure records used for fine-tuning')
        a.set_xticks(sorted(df.n.unique()))

    # legend outside the axes so it never covers data
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5),
               title='solid = fails\ncross-subject\n\ndashed = control',
               title_fontsize=7.5, frameon=False)
    fig.suptitle('Patient calibration recovers the subjects a population model fails on',
                 y=1.04, fontsize=11)
    fig.tight_layout()
    _save(fig, fname)
    return fig


# =============================================================================
# Figure 2 — LOSO bimodality
# =============================================================================

def fig2_loso(per_fold, model='logreg', fname='fig2_loso_bimodality'):
    """per_fold from run_loso / loso_folds.csv. Shows the distribution a mean hides."""
    d = per_fold[per_fold.model == model].copy() if 'model' in per_fold else per_fold.copy()
    d = d.sort_values('auprc_lift')

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8),
                           gridspec_kw={'width_ratios': [2.3, 1]})

    cols = ['#c0392b' if v < 5 else '#2c3e50' for v in d.auprc_lift]
    ax[0].barh(range(len(d)), d.auprc_lift, color=cols, height=0.72)
    ax[0].set_yticks(range(len(d)))
    ax[0].set_yticklabels(d.held_out, fontsize=7.5)
    ax[0].set_xscale('log')
    ax[0].axvline(1.0, color='#666', ls=':', lw=1)
    ax[0].axvline(d.auprc_lift.median(), color='#27ae60', ls='--', lw=1.2)
    ax[0].text(d.auprc_lift.median() * 1.08, 0.4,
               f'median {d.auprc_lift.median():.1f}x', color='#27ae60', fontsize=8)
    ax[0].set_xlabel('AUPRC lift over baseline (log scale)')
    ax[0].set_title(f'Leave-one-subject-out, {model} — {(d.auprc_lift < 5).sum()}'
                    f'/{len(d)} folds at chance', loc='left')

    ax[1].scatter(d.auprc_lift, d.sensitivity_event, c=cols, s=34, zorder=3)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('AUPRC lift')
    ax[1].set_ylabel('event sensitivity')
    ax[1].axvline(5, color='#c0392b', ls=':', lw=1)
    ax[1].set_title('no middle ground', loc='left')

    fig.tight_layout()
    _save(fig, fname)
    return fig


# =============================================================================
# Figure 3 — threshold transfer
# =============================================================================

def fig3_threshold(cache, target_fa=2.0, max_duty=0.01,
                   fname='fig3_threshold_transfer'):
    """Needs `cache`, `evaluate`, `select_quantile`, `select_threshold` in scope."""
    a = evaluate(cache, target_fa, max_duty, use_quantile=False)
    b = evaluate(cache, target_fa, max_duty, use_quantile=True)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))

    pos = np.arange(len(a))
    ax[0].bar(pos - 0.2, a.duty, 0.4, label='absolute threshold', color='#c0392b')
    ax[0].bar(pos + 0.2, b.duty, 0.4, label='quantile transfer', color='#2980b9')
    ax[0].axhline(max_duty, color='#333', ls='--', lw=1)
    ax[0].text(0.01, max_duty * 1.15, f'calibration target {max_duty}',
               fontsize=7.5, transform=ax[0].get_yaxis_transform())
    ax[0].set_yscale('log')
    ax[0].set_ylabel('alarm duty on TEST (log)')
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels([f'{s[3:]}·{n}' for s, n in zip(a.subject, a.n)],
                          rotation=90, fontsize=6.5)
    ax[0].set_xlabel('subject · n')
    ax[0].legend()
    ax[0].set_title('Absolute thresholds do not survive session change', loc='left')

    ax[1].scatter(a.fa, a.sens, s=36, color='#c0392b', label='absolute', zorder=3)
    ax[1].scatter(b.fa, b.sens, s=36, color='#2980b9', marker='s',
                  label='quantile', zorder=3)
    ax[1].set_xlabel('false alarms per hour')
    ax[1].set_ylabel('event sensitivity')
    ax[1].set_xscale('symlog', linthresh=1)
    ax[1].legend()
    ax[1].set_title('quantile transfer keeps FA/h usable', loc='left')

    fig.tight_layout()
    _save(fig, fname)
    print(f'  absolute: median duty {a.duty.median():.4f}, max {a.duty.max():.4f}')
    print(f'  quantile: median duty {b.duty.median():.4f}, max {b.duty.max():.4f}')
    return fig


# =============================================================================
# Figure 4 — corpus composition (supplementary)
# =============================================================================

def fig4_corpus(m, fname='fig4_corpus'):
    """m from the 8.2 verification table: shard, windows, ictal, events."""
    d = m.copy()
    d['w_per_ev'] = d.ictal / d.events.replace(0, np.nan)
    d = d.sort_values('events', ascending=False)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    ax[0].bar(range(len(d)), d.events, color='#2c3e50')
    ax[0].set_xticks(range(len(d)))
    ax[0].set_xticklabels(d.shard, rotation=90, fontsize=6.5)
    ax[0].set_ylabel('annotated seizure events')
    ax[0].set_title(f'{int(d.events.sum())} events, 44% from four subjects', loc='left')

    ax[1].bar(range(len(d)), d.w_per_ev, color='#8e44ad')
    ax[1].set_xticks(range(len(d)))
    ax[1].set_xticklabels(d.shard, rotation=90, fontsize=6.5)
    ax[1].set_ylabel('ictal windows per event')
    ax[1].set_title('seizure duration spans 29x', loc='left')

    pct = d.ictal / d.windows * 100
    ax[2].bar(range(len(d)), pct, color='#c0392b')
    ax[2].axhline(0.339, color='#333', ls='--', lw=1)
    ax[2].text(0.02, 0.36, 'corpus 0.339%', fontsize=7.5,
               transform=ax[2].get_yaxis_transform())
    ax[2].set_xticks(range(len(d)))
    ax[2].set_xticklabels(d.shard, rotation=90, fontsize=6.5)
    ax[2].set_ylabel('ictal windows (%)')
    ax[2].set_title('class imbalance by subject', loc='left')

    fig.tight_layout()
    _save(fig, fname)
    return fig


print('figure functions loaded: fig1_calibration, fig2_loso, fig3_threshold, fig4_corpus')
