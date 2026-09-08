#!/usr/bin/env python3
"""
CHB-MIT low-latency seizure decoder — full pipeline, headless.

Consolidates every validated stage from the notebook into one runnable module.
Each stage is idempotent and cached, so re-running is cheap and a crash costs
one subject rather than the whole corpus.

    python chbmit_pipeline.py --stage manifest
    python chbmit_pipeline.py --stage extract
    python chbmit_pipeline.py --stage extract --subjects chb01 chb17a
    python chbmit_pipeline.py --stage verify
    python chbmit_pipeline.py --stage selftest

Expected corpus totals (all asserted by --stage verify):
    686 EDFs | 676 annotated | 198 seizures | 3.23 h ictal
    683 native_bipolar / 1 rederived_cs2 / 2 rederived_bare
    3,480,880 windows | 11,809 ictal (0.339%) | 23 subject groups
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pywt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

# ============================================================================
# Configuration
# ============================================================================

ROOT = os.environ.get('CHBMIT_ROOT', '/content/chbmit')
OUT = os.environ.get('CHBMIT_OUT', '/content/data')
FEATURES = f'{OUT}/features'

SFREQ = 256.0
WINDOW_SEC, STRIDE_SEC, GUARD_SEC = 2.0, 1.0, 60.0
L_FREQ, H_FREQ, FILTER_ORDER = 0.5, 40.0, 2

# ============================================================================
# Channels
# ============================================================================
# Index order is FIXED and load-bearing: downstream code indexes channels by
# integer position. Reordering silently corrupts everything after it.
#
# Dropped from the reference 23-channel montage:
#   T7-FT9 / FT9-FT10 / FT10-T8 -- FT9/FT10 are not standard 10-20 positions
#     and the chain is inconsistently present, which would make the feature
#     vector's shape subject-dependent.
#   P7-T7 -- polarity inverse of T7-P7; including both gives a collinear pair.

CANONICAL = (
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',      # left temporal
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',      # left parasagittal
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',      # right parasagittal
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',      # right temporal
    'FZ-CZ', 'CZ-PZ',                          # midline
)
N_CHANNELS = len(CANONICAL)

ALIASES = {'01': 'O1'}          # zero-one typo in the chb12_28/_29 EDF headers
SUBJECT_ALIASES = {'chb21': 'chb01'}   # same patient, recorded 1.5 years apart
NOISY_RECORDS = {'chb12_27.edf'}       # session-specific narrowband interference
KNOWN_UNANNOTATED = {f'chb24_{n:02d}.edf'
                     for n in (2, 5, 8, 10, 12, 16, 18, 19, 20, 22)}


def norm(name: str) -> str:
    """Uppercase, strip MNE's duplicate-run suffix, apply aliases.

    'T8-P8-0' -> 'T8-P8'   (MNE disambiguates the duplicate in the header)
    'FT9-FT10' -> unchanged (must NOT lose its trailing digits)
    '01' -> 'O1'           (recording-time typo, not a real label)
    """
    n = name.strip().upper().replace(' ', '')
    m = re.search(r'-(\d+)$', n)
    if m and n[:m.start()].count('-') >= 1:
        n = n[:m.start()]
    return ALIASES.get(n, n)


def subject_group_of(record: str) -> str:
    """CV grouping key. Folds chb21->chb01 and chb17a/b/c->chb17.

    LEAKAGE TRAP: grouping by directory name puts the same person on both sides
    of a split. chb21 is chb01 1.5 years later; chb17a/b/c share one summary.
    """
    s = record.split('_')[0]
    s = SUBJECT_ALIASES.get(s, s)
    return 'chb17' if s.startswith('chb17') else s


# ============================================================================
# Summary parsing
# ============================================================================
# Regex-anchored on the NUMERIC VALUE, not on a ': ' delimiter. chb12_27 mixes
# one- and two-space separators within a single record; a delimiter split
# returns ' 951 seconds', and .split(' ')[0] on a leading space yields ''.
# That silently dropped 12 seizures.

RE_FILE = re.compile(r'^File\s+Name:\s*(\S+)', re.I)
RE_NSZ = re.compile(r'^Number\s+of\s+Seizures\s+in\s+File:\s*(\d+)', re.I)
RE_START = re.compile(r'^Seizure\s*\d*\s*Start\s+Time:\s*([\d.]+)\s*seconds', re.I)
RE_END = re.compile(r'^Seizure\s*\d*\s*End\s+Time:\s*([\d.]+)\s*seconds', re.I)


class EDFFileSummary:
    __slots__ = ('filename', 'n_declared', 'seizures')

    def __init__(self, filename):
        self.filename = filename
        self.n_declared = 0
        self.seizures = []

    def __repr__(self):
        return f'<{self.filename}: {len(self.seizures)} seizures>'


def parse_summary(path):
    """Parse one chbNN-summary.txt. RAISES on a declared/parsed count mismatch.

    The assertion is the load-bearing part, not the regex. The original bug
    emitted warnings, which scroll off the top of a Colab output, and the
    corrupted annotations propagated into a measurement that looked significant.
    """
    summaries, cur, pending = {}, None, None

    for lineno, raw in enumerate(open(path, errors='replace'), 1):
        line = raw.strip()
        if not line:
            continue

        if m := RE_FILE.match(line):
            if pending is not None:
                raise ValueError(f'{path}:{lineno} unterminated seizure in {cur.filename}')
            cur = EDFFileSummary(m.group(1))
            summaries[cur.filename] = cur
        elif cur is None:
            continue                              # header before the first record
        elif m := RE_NSZ.match(line):
            cur.n_declared = int(m.group(1))
        elif m := RE_START.match(line):
            if pending is not None:
                raise ValueError(f'{path}:{lineno} two starts in a row, {cur.filename}')
            pending = float(m.group(1))
        elif m := RE_END.match(line):
            end = float(m.group(1))
            if pending is None:
                raise ValueError(f'{path}:{lineno} end with no start, {cur.filename}')
            if end < pending:
                raise ValueError(f'{path}:{lineno} end {end} < start {pending}')
            cur.seizures.append((pending, end))
            pending = None
        # 'Channels changed:' blocks and 'Channel N:' lines fall through

    if pending is not None:
        raise ValueError(f'{path}: file ends mid-seizure in {cur.filename}')

    for rec in summaries.values():
        if len(rec.seizures) != rec.n_declared:
            raise ValueError(f'{rec.filename}: declared {rec.n_declared}, '
                             f'parsed {len(rec.seizures)}')
    return summaries


def build_summaries(root=ROOT):
    """Parse every summary into one dict. Asserts the corpus total."""
    out = {}
    for s in sorted(glob.glob(f'{root}/chb*/chb*-summary.txt')):
        out.update(parse_summary(s))
    if len(out) != 676:
        raise RuntimeError(f'expected 676 annotated records, got {len(out)}')
    return out


# ============================================================================
# Loader
# ============================================================================

def load_18(path, summaries=None):
    """Return (data, derivation): (18, n_samples) in MICROVOLTS, canonical order.

    Mode check is ORDERED -- native, then CS2, then bare -- so a recorded signal
    always beats a derived one.

    Units: microvolts, not volts. In volts, EEG variance is ~3e-9 and a fixed
    1e-12 epsilon guard becomes a ~1% distortion on Hjorth mobility. See the
    _EPS note in FeatureExtractor.

    Raises rather than returning a partial or misordered array.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    lut = {}
    for i, c in enumerate(raw.ch_names):
        lut.setdefault(norm(c), i)        # first occurrence wins (T8-P8-0 over -1)
    D = raw.get_data()
    pairs = [p.split('-') for p in CANONICAL]

    if all(c in lut for c in CANONICAL):
        out, mode = D[[lut[c] for c in CANONICAL]], 'native_bipolar'

    elif all(f'{a}-CS2' in lut and f'{b}-CS2' in lut for a, b in pairs):
        # shared CS2 reference cancels identically -> exact
        out = np.empty((N_CHANNELS, D.shape[1]))
        for k, (a, b) in enumerate(pairs):
            out[k] = D[lut[f'{a}-CS2']] - D[lut[f'{b}-CS2']]
        mode = 'rederived_cs2'

    elif all(a in lut and b in lut for a, b in pairs):
        # bare labels: a common reference is ASSUMED, not stated in the header
        out = np.empty((N_CHANNELS, D.shape[1]))
        for k, (a, b) in enumerate(pairs):
            out[k] = D[lut[a]] - D[lut[b]]
        mode = 'rederived_bare'

    else:
        missing = [c for c in CANONICAL if c not in lut]
        raise ValueError(f'{path}: no derivation path; missing {missing[:4]}')

    out = out * 1e6                                    # volts -> microvolts
    p99 = np.percentile(np.abs(out), 99)
    if not 1.0 < p99 < 1e4:                            # scalp EEG is ~20-200 uV
        raise ValueError(f'{path} [{mode}]: p99 {p99:.1f} uV is not scalp EEG')
    return out, mode


# ============================================================================
# Causal filtering
# ============================================================================

class CausalBandpass:
    """Forward-only Butterworth bandpass with persistent per-channel state.

    ORDER: for a bandpass design scipy's `order` yields realised order 2*order.
    order=2 here is a 4th-order bandpass in 2 SOS sections.

    Stream equivalence is BIT-IDENTICAL: filtering one pass over a record equals
    streaming it in 4-sample packets with zi carried (verified 0.000e+00).
    Never filtfilt -- non-causal, leaks future samples.
    """

    def __init__(self, sfreq=SFREQ, l_freq=L_FREQ, h_freq=H_FREQ,
                 order=FILTER_ORDER, n_channels=N_CHANNELS):
        nyq = sfreq / 2.0
        if not 0 < l_freq < h_freq < nyq:
            raise ValueError(f'bad band {l_freq}-{h_freq} Hz for sfreq={sfreq}')
        self.sfreq, self.n_channels = float(sfreq), int(n_channels)
        self.sos = signal.butter(order, [l_freq / nyq, h_freq / nyq],
                                 btype='band', output='sos')
        self._zi_proto = signal.sosfilt_zi(self.sos)        # (n_sections, 2)
        self.reset()

    def reset(self, primer=None):
        """Clear state. `primer` (n_channels,) initialises to steady state for
        that DC level. Measured: 27x less startup garbage on a DC-offset signal.
        Call PER RECORD -- state must never cross recording boundaries."""
        zi = np.repeat(self._zi_proto[:, None, :], self.n_channels, axis=1)
        self._zi = zi * (np.asarray(primer, float)[None, :, None]
                         if primer is not None else 0.0)

    def process(self, block):
        block = np.asarray(block, dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != self.n_channels:
            raise ValueError(f'expected ({self.n_channels}, n), got {block.shape}')
        out, self._zi = signal.sosfilt(self.sos, block, axis=-1, zi=self._zi)
        return out

    def group_delay_ms(self, freqs):
        b, a = signal.sos2tf(self.sos)
        _, gd = signal.group_delay((b, a), w=np.asarray(freqs, float), fs=self.sfreq)
        return gd / self.sfreq * 1e3


# ============================================================================
# Windowing and labelling
# ============================================================================

LAB_ICTAL, LAB_INTER, LAB_DROP = 1, 0, -1


def label_windows(starts, window_sec, seizures, guard_sec):
    """Returns (labels, event_idx).

    Ictal is applied BEFORE the guard band, so ictal wins: a window straddling
    one seizure's guard and the next seizure's onset is correctly ictal.

    Guard-band windows are DROPPED, not labelled interictal. A window ending 3 s
    before onset is not a clean negative, and CHB-MIT onset marks are reviewer
    estimates.
    """
    ends = starts + window_sec
    lab = np.full(starts.shape, LAB_INTER, dtype=np.int8)
    ev = np.full(starts.shape, -1, dtype=np.int16)

    for k, (s0, s1) in enumerate(seizures):
        ov = (starts < s1) & (ends > s0)
        lab[ov] = LAB_ICTAL
        ev[ov] = k

    for s0, s1 in seizures:
        near = (starts < s1 + guard_sec) & (ends > s0 - guard_sec)
        lab[near & (lab != LAB_ICTAL)] = LAB_DROP

    return lab, ev


def window_record(path, summary, summaries=None, sfreq=SFREQ,
                  window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC,
                  guard_sec=GUARD_SEC):
    """Load, filter causally, window, and label one record."""
    record = os.path.basename(path)

    if summary is None:
        # Distinguish 'genuinely unannotated' from 'you forgot to build the dict'.
        # An empty summaries dict once produced 3 silently-empty shards.
        if summaries is not None and not summaries:
            raise RuntimeError('summaries dict is empty -- rebuild it')
        if record not in KNOWN_UNANNOTATED:
            raise ValueError(f'{record}: no summary entry and not a known '
                             f'unannotated record')
        return None

    X, derivation = load_18(path)

    bp = CausalBandpass(sfreq=sfreq, n_channels=N_CHANNELS)
    bp.reset(primer=X[:, 0])
    X = bp.process(X)

    # Parameterize by DURATION, not sample count: frequency resolution depends
    # on window duration, so a 2 s window gives 0.5 Hz bins at 256 and 128 Hz
    # alike. Hardcoding 512 makes the sampling-rate decision expensive.
    w_size = int(round(window_sec * sfreq))
    w_stride = int(round(stride_sec * sfreq))
    if X.shape[1] < w_size:
        return None

    # sliding_window_view returns a VIEW, not a copy -- avoids a 2x blowup at
    # 50% overlap, but the windows alias X, so do not mutate in place.
    W = sliding_window_view(X, w_size, axis=-1)[:, ::w_stride, :].transpose(1, 0, 2)

    starts = np.arange(W.shape[0], dtype=np.float64) * stride_sec
    lab, ev = label_windows(starts, window_sec, summary.seizures, guard_sec)

    keep = lab != LAB_DROP
    if not keep.any():
        return None

    meta = {
        'y': lab[keep], 't_start': starts[keep], 'record': record,
        'subject': record.split('_')[0],
        'subject_group': subject_group_of(record),
        'seizure_event': [f'{record}#{e}' if e >= 0 else '' for e in ev[keep]],
        'derivation': derivation,
        'noisy': record in NOISY_RECORDS,
    }
    return W[keep], lab[keep], meta


# ============================================================================
# Feature extraction
# ============================================================================

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 40)}

# _EPS is safe ONLY because load_18 returns microvolts. In volts, EEG variance
# is ~3e-9 and this epsilon becomes a ~1% distortion on Hjorth mobility -- a
# bug that passes every shape, finiteness, and NaN check and is visible only
# under differential testing against an independent implementation.
_EPS = 1e-12

FAMILIES = ('hjorth', 'linelength', 'dwt', 'bandpower', 'spec_entropy', 'aperiodic')


class FeatureExtractor:
    """Precomputed, vectorized extractor for fixed-shape EEG windows.

    Everything constant is built once at construction: the taper, the frequency
    masks, and the log-log design-matrix pseudo-inverse. The hot path allocates
    only its output and never loops over channels in Python.
    """

    def __init__(self, sfreq=SFREQ, n_channels=N_CHANNELS, window_size=512,
                 families=FAMILIES, fit_lo=1.0, fit_hi=40.0,
                 wavelet='db4', dwt_level=4):
        self.sfreq, self.n_channels, self.window_size = sfreq, n_channels, window_size
        self.families, self.wavelet, self.dwt_level = tuple(families), wavelet, dwt_level

        self._taper = np.hanning(window_size)
        self._tnorm = float((self._taper ** 2).sum())
        self._freqs = np.fft.rfftfreq(window_size, 1 / sfreq)
        self._masks = {n: (self._freqs >= lo) & (self._freqs < hi)
                       for n, (lo, hi) in BANDS.items()}
        self._support = (self._freqs >= 1.0) & (self._freqs < 40.0)
        self._log2n = float(np.log2(max(int(self._support.sum()), 2)))

        # Aperiodic slope: the design matrix depends only on the frequency grid,
        # so its pseudo-inverse is a CONSTANT. This is what makes the exponent
        # real-time viable (0.002 ms vs 0.623 ms for a polyfit loop, agreeing to
        # 8.8e-15). FOOOF's non-linear fit is the slow part, not the quantity.
        fm = (self._freqs >= fit_lo) & (self._freqs <= fit_hi) & (self._freqs > 0)
        logf = np.log10(self._freqs[fm])
        self._fit_mask = fm
        self._pinv = np.linalg.pinv(np.column_stack([np.ones_like(logf), logf]))

        if 'dwt' in self.families:
            mx = pywt.dwt_max_level(window_size, pywt.Wavelet(wavelet).dec_len)
            if dwt_level > mx:
                raise ValueError(f'dwt_level {dwt_level} > max {mx}')

        self.feature_names = self._names()
        self.n_features = len(self.feature_names)

    def _suffixes(self):
        s = []
        if 'hjorth' in self.families:
            s += ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
        if 'linelength' in self.families:
            s += ['linelength']
        if 'dwt' in self.families:
            s += [f'dwt_cD{i}' for i in range(1, self.dwt_level + 1)]
            s += [f'dwt_cA{self.dwt_level}']
        if 'bandpower' in self.families:
            s += [f'relpow_{b}' for b in BANDS]
        if 'spec_entropy' in self.families:
            s += ['spec_entropy']
        if 'aperiodic' in self.families:
            s += ['aperiodic_slope']
        return s

    def _names(self):
        chans = (list(CANONICAL) if self.n_channels == N_CHANNELS
                 else [f'ch{i:02d}' for i in range(self.n_channels)])
        return ([f'{c}__{s}' for c in chans for s in self._suffixes()]
                + ['guard_frontal', 'guard_temporal', 'guard_global'])

    def transform_batch(self, X):
        """(n_win, n_ch, window_size) -> (n_win, n_features) float32."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        blocks = []

        if any(f in self.families for f in ('bandpower', 'spec_entropy', 'aperiodic')):
            spec = np.fft.rfft(X * self._taper, axis=-1)
            psd = (spec.real ** 2 + spec.imag ** 2) / self._tnorm + _EPS

        if 'hjorth' in self.families:
            d1, d2 = np.diff(X, axis=-1), np.diff(X, n=2, axis=-1)
            v0, v1, v2 = X.var(-1) + _EPS, d1.var(-1) + _EPS, d2.var(-1) + _EPS
            mob = np.sqrt(v1 / v0)
            blocks.append(np.stack([np.log10(v0), mob, np.sqrt(v2 / v1) / mob], -1))

        if 'linelength' in self.families:
            ll = np.mean(np.abs(np.diff(X, axis=-1)), axis=-1)
            blocks.append(np.log10(ll + _EPS)[..., None])

        if 'dwt' in self.families:
            c = pywt.wavedec(X, self.wavelet, level=self.dwt_level, axis=-1)
            # wavedec returns [cA_n, cD_n, ..., cD1] -- REVERSE frequency order.
            # Naming these in list order silently mislabels every band.
            details = c[1:][::-1]                        # now cD1 .. cD_n
            e = [np.log10((d ** 2).mean(-1) + _EPS) for d in details]
            e.append(np.log10((c[0] ** 2).mean(-1) + _EPS))
            blocks.append(np.stack(e, -1))

        if 'bandpower' in self.families:
            tot = psd[..., self._support].sum(-1, keepdims=True) + _EPS
            blocks.append(np.stack([psd[..., m].sum(-1) / tot[..., 0]
                                    for m in self._masks.values()], -1))

        if 'spec_entropy' in self.families:
            p = psd[..., self._support]
            p = p / (p.sum(-1, keepdims=True) + _EPS)
            blocks.append((-(p * np.log2(p + _EPS)).sum(-1) / self._log2n)[..., None])

        if 'aperiodic' in self.families:
            blocks.append((np.log10(psd[..., self._fit_mask]) @ self._pinv.T)[..., 1:2])

        flat = np.concatenate(blocks, -1).reshape(X.shape[0], -1)

        front = [CANONICAL.index(c) for c in ('FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8')]
        temp = [CANONICAL.index(c) for c in ('F7-T7', 'T7-P7', 'F8-T8', 'T8-P8')]
        guards = np.stack([
            np.log10(X[:, front].var(axis=(1, 2)) + _EPS),
            np.log10(np.diff(X[:, temp], axis=-1).var(axis=(1, 2)) + _EPS),
            np.log10(X.var(-1).mean(-1) + _EPS)], -1)

        return np.concatenate([flat, guards], -1).astype(np.float32)


# ============================================================================
# Evaluation
# ============================================================================

def window_metrics(y_true, scores):
    """Threshold-free ranking metrics. AUPRC is meaningless without its baseline.

    At a 0.34% positive rate a useless model scores AUPRC ~0.0034. Read
    auprc_lift, not auprc: 0.05 is a 15x lift and genuinely good.

    ACCURACY IS NOT REPORTED ANYWHERE. A null model scores 99.66%.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    pos_rate = float(y_true.mean())
    out = {'positive_rate': pos_rate, 'auprc_baseline': pos_rate}
    if len(np.unique(y_true)) < 2:
        return {**out, 'auroc': np.nan, 'auprc': np.nan, 'auprc_lift': np.nan}
    out['auroc'] = float(roc_auc_score(y_true, scores))
    out['auprc'] = float(average_precision_score(y_true, scores))
    out['auprc_lift'] = out['auprc'] / pos_rate
    return out


def report_folds(per_fold, metrics=('auprc', 'auprc_lift', 'auroc')):
    """Print PER-FOLD results, then the summary. Never the mean alone.

    Positives are extremely concentrated in this corpus: chb12 (40 events),
    chb15 (20), chb24 (16) and chb13 (12) hold 88 of 198 -- 44% from four
    subjects -- while chb02/07/11/19/22 have 3 each. Fold variance is large and
    a mean hides it. Report the spread and the worst fold, always.
    """
    df = pd.DataFrame(per_fold)
    print(df.to_string(index=False))
    print()
    for m in metrics:
        if m not in df:
            continue
        v = df[m].dropna()
        if not len(v):
            continue
        worst = df.loc[v.idxmin()]
        print(f'{m:12s} mean {v.mean():7.4f}  median {v.median():7.4f}  '
              f'min {v.min():7.4f}  max {v.max():7.4f}  '
              f'(worst fold: {worst.get("held_out", "?")})')
    return df


# ============================================================================
# Stages
# ============================================================================

def stage_manifest(args):
    """Per-record inclusion decisions -> record_manifest.csv."""
    summaries = build_summaries(args.root)
    rows = []
    for p in sorted(glob.glob(f'{args.root}/chb*/*.edf')):
        name = os.path.basename(p)
        rec = summaries.get(name)
        rows.append({
            'record': name,
            'subject_dir': name.split('_')[0],
            'subject_group': subject_group_of(name),
            'annotated': rec is not None,
            'n_seizures': len(rec.seizures) if rec else None,
            'ictal_sec': sum(e - s for s, e in rec.seizures) if rec else None,
            'size_mb': round(os.path.getsize(p) / 1e6, 1),
        })
    man = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    man.to_csv(f'{args.out}/record_manifest.csv', index=False)
    print(f'{len(man)} records, {man.annotated.sum()} annotated, '
          f'{int(man.n_seizures.sum())} seizures, {man.ictal_sec.sum()/3600:.2f} h ictal')
    print(f'{man.subject_dir.nunique()} case dirs -> {man.subject_group.nunique()} groups')
    return man


def stage_extract(args):
    """Per-subject Parquet shards. Idempotent: cached subjects are skipped."""
    summaries = build_summaries(args.root)
    fx = FeatureExtractor(window_size=int(round(WINDOW_SEC * SFREQ)))
    os.makedirs(args.features, exist_ok=True)

    subjects = args.subjects or sorted(
        {os.path.basename(p).split('_')[0]
         for p in glob.glob(f'{args.root}/chb*/*.edf')})

    for subj in subjects:
        out_path = f'{args.features}/{subj}.parquet'
        if os.path.exists(out_path) and not args.force:
            print(f'{subj}: cached'); continue

        t0 = time.perf_counter()
        # Glob by RECORD PREFIX, not directory: chb17's files live in chb17/ but
        # carry chb17a_/b_/c_ prefixes. A directory glob finds nothing and the
        # loop reports 'no usable records', which looks like a data problem.
        paths = sorted(glob.glob(f'{args.root}/chb*/{subj}_*.edf'))
        if not paths:
            print(f'{subj}: 0 files found -- check the glob'); continue

        frames = []
        for p in paths:
            out = window_record(p, summaries.get(os.path.basename(p)), summaries)
            if out is None:
                continue
            W, y, meta = out
            feats = np.concatenate([fx.transform_batch(W[i:i + 2048])
                                    for i in range(0, len(W), 2048)])
            d = pd.DataFrame(feats, columns=fx.feature_names)
            for k in ('y', 't_start', 'seizure_event'):
                d.insert(0, k, meta[k])
            for k in ('noisy', 'derivation', 'subject_group', 'subject', 'record'):
                d.insert(0, k, meta[k])
            frames.append(d)
            del W, feats

        if not frames:
            print(f'{subj}: no usable records'); continue

        df = pd.concat(frames, ignore_index=True)
        # Categorical dtypes on the low-cardinality string columns: 211 -> 171 MB
        # on chb01, which is the float32 floor.
        for c in ('record', 'subject', 'subject_group', 'derivation', 'seizure_event'):
            df[c] = df[c].astype('category')
        df.to_parquet(out_path, index=False)
        if args.drive:
            os.makedirs(args.drive, exist_ok=True)
            df.to_parquet(f'{args.drive}/{subj}.parquet', index=False)
        print(f'{subj}: {len(df):,} windows, {(df.y == 1).sum():,} ictal, '
              f'{time.perf_counter() - t0:.0f}s')
        del df, frames
        gc.collect()


def stage_verify(args):
    """Assert corpus totals. The check that catches silent seizure loss.

    Reads COLUMNS ONLY. pd.read_parquet on the whole features directory loads
    ~4 GB before pandas overhead and has crashed the runtime.
    """
    rows = []
    for f in sorted(glob.glob(f'{args.features}/*.parquet')):
        d = pd.read_parquet(f, columns=['y', 'subject_group', 'seizure_event'])
        rows.append({'shard': os.path.basename(f)[:-8],
                     'group': str(d.subject_group.iloc[0]),
                     'windows': len(d), 'ictal': int((d.y == 1).sum()),
                     'events': d.loc[d.y == 1, 'seizure_event'].nunique()})
        del d
    m = pd.DataFrame(rows)
    print(m.to_string(index=False))
    print(f'\nwindows {m.windows.sum():,} | ictal {m.ictal.sum():,} '
          f'({m.ictal.sum() / m.windows.sum() * 100:.3f}%)')
    print(f'events {m.events.sum()} (expect 198) | '
          f'groups {m.group.nunique()} (expect 23)')

    assert m.events.sum() == 198, f'SEIZURE LOSS: {m.events.sum()} events, expected 198'
    assert m.group.nunique() == 23, f'grouping wrong: {m.group.nunique()} groups'
    print('\nPASS')
    return m


def stage_selftest(args):
    """Validate the pipeline with no data access."""
    ok = True

    assert norm('T8-P8-0') == 'T8-P8'
    assert norm('FT9-FT10') == 'FT9-FT10'
    assert norm('01') == 'O1'
    print('PASS  normalizer')

    assert subject_group_of('chb21_19.edf') == 'chb01'
    assert subject_group_of('chb17a_03.edf') == 'chb17'
    assert subject_group_of('chb01_03.edf') == 'chb01'
    print('PASS  subject grouping')

    starts = np.arange(0, 200, 1.0)
    lab, ev = label_windows(starts, 2.0, [(100., 110.)], 20.)
    assert starts[lab == 1].min() == 99.0 and starts[lab == 1].max() == 109.0
    assert starts[lab == -1].min() == 79.0 and starts[lab == -1].max() == 129.0
    lab2, _ = label_windows(starts, 2.0, [(50., 55.), (70., 75.)], 20.)
    assert lab2[starts == 69.0][0] == LAB_ICTAL, 'ictal must win over the guard band'
    label_windows(starts, 2.0, [(0., 10.)], 20.)          # negative guard, no crash
    print('PASS  window labelling (incl. ictal-over-guard and t=0 seizure)')

    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_CHANNELS, 4096)) * 100.0
    f1 = CausalBandpass(); whole = f1.process(x)
    f2 = CausalBandpass()
    streamed = np.concatenate([f2.process(x[:, i:i + 4])
                               for i in range(0, 4096, 4)], axis=-1)
    diff = np.abs(whole - streamed).max()
    assert diff < 1e-12, f'state not carried across blocks: {diff:.3e}'
    naive = np.concatenate([signal.sosfilt(f1.sos, x[:, i:i + 4], axis=-1)
                            for i in range(0, 4096, 4)], axis=-1)
    assert np.abs(whole - naive).max() > 1e-3, 'negative control failed'
    print(f'PASS  stream equivalence (max abs diff {diff:.3e})')

    fx = FeatureExtractor()
    assert fx.n_features == 291, fx.n_features
    W = rng.standard_normal((32, N_CHANNELS, 512)) * 100.0
    F = fx.transform_batch(W)
    assert F.shape == (32, 291) and np.isfinite(F).all()
    print(f'PASS  feature extractor ({fx.n_features} features, all finite)')

    # differential test: the check that caught the microvolt epsilon bug
    w0 = W[0]
    manual = np.sqrt(np.diff(w0, axis=-1).var(-1) / w0.var(-1))
    idx = [fx.feature_names.index(f'{c}__hjorth_mobility') for c in CANONICAL]
    d = np.abs(F[0][idx] - manual).max()
    assert d < 1e-6, f'hjorth mobility differs by {d:.3e}'
    print(f'PASS  differential test, hjorth mobility (diff {d:.2e}, float32 floor)')

    y = (rng.random(100_000) < 0.0034).astype(int)
    assert window_metrics(y, y + rng.random(len(y)) * 1e-9)['auprc'] > 0.99
    assert 0.7 < window_metrics(y, rng.random(len(y)))['auprc_lift'] < 1.4
    assert np.isnan(window_metrics(np.zeros(100, int), rng.random(100))['auprc'])
    print('PASS  window_metrics (perfect / random / single-class)')

    print('\nALL SELF-TESTS PASS' if ok else '\nFAILURES')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True,
                    choices=['manifest', 'extract', 'verify', 'selftest'])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--features', default=FEATURES)
    ap.add_argument('--drive', default=None,
                    help='optional second write target, e.g. Drive features dir')
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--force', action='store_true', help='re-extract cached shards')
    args = ap.parse_args()

    {'manifest': stage_manifest, 'extract': stage_extract,
     'verify': stage_verify, 'selftest': stage_selftest}[args.stage](args)


if __name__ == '__main__':
    sys.exit(main())
