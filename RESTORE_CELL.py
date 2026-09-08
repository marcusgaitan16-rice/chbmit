# ============================================================================
# RESTORE CELL — paste at the top of the notebook, run first after any restart
# ============================================================================
# Colab wipes Python state on restart but keeps /content unless the VM was
# recycled. This cell rebuilds every symbol the pipeline needs and asserts that
# each one is correct. Takes ~30 s.
#
# It does NOT re-download, re-extract, or re-fit anything.
# ============================================================================

# --- 0. stdlib + scientific -------------------------------------------------
import os, re, glob, time, gc, json, shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import matplotlib.pyplot as plt

try:
    import mne, pywt
except ImportError:
    raise SystemExit('run:  !pip install -q mne PyWavelets pyarrow  then RESTART')
mne.set_log_level('ERROR')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT, OUT = '/content/chbmit', '/content/data'
FEATURES = f'{OUT}/features'
DRIVE = '/content/drive/MyDrive/chbmit'

# --- 1. what survived? ------------------------------------------------------
n_edf = len(glob.glob(f'{ROOT}/chb*/*.edf'))
n_shard = len(glob.glob(f'{FEATURES}/*.parquet'))
print(f'{n_edf:4d} EDFs        (686 expected; 0 -> VM recycled, re-sync from S3)')
print(f'{n_shard:4d} shards      (26 expected; 0 -> restore from Drive)')

# --- 2. channel constants ---------------------------------------------------
# Index order is FIXED and load-bearing: downstream code indexes by position.
CANONICAL = ('FP1-F7','F7-T7','T7-P7','P7-O1',    # left temporal
             'FP1-F3','F3-C3','C3-P3','P3-O1',    # left parasagittal
             'FP2-F4','F4-C4','C4-P4','P4-O2',    # right parasagittal
             'FP2-F8','F8-T8','T8-P8','P8-O2',    # right temporal
             'FZ-CZ','CZ-PZ')                     # midline
N_CHANNELS = len(CANONICAL)

ALIASES = {'01': 'O1'}                     # zero-one typo in chb12_28/_29 headers
SUBJECT_ALIASES = {'chb21': 'chb01'}       # same patient, 1.5 years apart
NOISY_RECORDS = {'chb12_27.edf'}           # session-specific interference
KNOWN_UNANNOTATED = {f'chb24_{n:02d}.edf'
                     for n in (2,5,8,10,12,16,18,19,20,22)}


def norm(name):
    """Strip MNE's duplicate-run suffix without eating FT9-FT10; apply aliases."""
    n = name.strip().upper().replace(' ', '')
    m = re.search(r'-(\d+)$', n)
    if m and n[:m.start()].count('-') >= 1:
        n = n[:m.start()]
    return ALIASES.get(n, n)


def subject_group_of(record):
    """CV grouping key. Folds chb21->chb01 and chb17a/b/c->chb17."""
    s = record.split('_')[0]
    s = SUBJECT_ALIASES.get(s, s)
    return 'chb17' if s.startswith('chb17') else s


# --- 3. summary parser ------------------------------------------------------
# Regex-anchored on the NUMERIC VALUE. chb12_27 mixes one- and two-space
# separators within one record; a "': '" split silently dropped 12 seizures.
RE_FILE  = re.compile(r'^File\s+Name:\s*(\S+)', re.I)
RE_NSZ   = re.compile(r'^Number\s+of\s+Seizures\s+in\s+File:\s*(\d+)', re.I)
RE_START = re.compile(r'^Seizure\s*\d*\s*Start\s+Time:\s*([\d.]+)\s*seconds', re.I)
RE_END   = re.compile(r'^Seizure\s*\d*\s*End\s+Time:\s*([\d.]+)\s*seconds', re.I)


class EDFFileSummary:
    __slots__ = ('filename', 'n_declared', 'seizures')
    def __init__(self, filename):
        self.filename, self.n_declared, self.seizures = filename, 0, []
    def __repr__(self):
        return f'<{self.filename}: {len(self.seizures)} seizures>'


def parse_summary(path):
    """RAISES on a declared/parsed seizure-count mismatch. The assertion is the
    load-bearing part — warnings scroll off the top of a Colab output."""
    summaries, cur, pending = {}, None, None
    for lineno, raw in enumerate(open(path, errors='replace'), 1):
        line = raw.strip()
        if not line:
            continue
        if m := RE_FILE.match(line):
            if pending is not None:
                raise ValueError(f'{path}:{lineno} unterminated seizure')
            cur = EDFFileSummary(m.group(1)); summaries[cur.filename] = cur
        elif cur is None:
            continue
        elif m := RE_NSZ.match(line):
            cur.n_declared = int(m.group(1))
        elif m := RE_START.match(line):
            if pending is not None:
                raise ValueError(f'{path}:{lineno} two starts in a row')
            pending = float(m.group(1))
        elif m := RE_END.match(line):
            end = float(m.group(1))
            if pending is None:
                raise ValueError(f'{path}:{lineno} end with no start')
            cur.seizures.append((pending, end)); pending = None
    if pending is not None:
        raise ValueError(f'{path}: ends mid-seizure')
    for rec in summaries.values():
        if len(rec.seizures) != rec.n_declared:
            raise ValueError(f'{rec.filename}: declared {rec.n_declared}, '
                             f'parsed {len(rec.seizures)}')
    return summaries


# --- 4. summaries dict ------------------------------------------------------
# An empty dict makes every record look unannotated and silently produces empty
# shards. This cost a full chb17 extraction once.
summarized = {}
if n_edf:
    for s in sorted(glob.glob(f'{ROOT}/chb*/chb*-summary.txt')):
        summarized.update(parse_summary(s))
    assert len(summarized) == 676, f'expected 676 records, got {len(summarized)}'


# --- 5. loader --------------------------------------------------------------
def load_18(path):
    """(18, n_samples) in MICROVOLTS, canonical order, plus the derivation tag.

    Mode check is ORDERED: a recorded signal always beats a derived one.
    Microvolts, not volts — in volts, EEG variance is ~3e-9 and a fixed 1e-12
    epsilon becomes a ~1% distortion on Hjorth mobility.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    lut = {}
    for i, c in enumerate(raw.ch_names):
        lut.setdefault(norm(c), i)              # first T8-P8 wins over the dup
    D = raw.get_data()
    pairs = [p.split('-') for p in CANONICAL]

    if all(c in lut for c in CANONICAL):
        out, mode = D[[lut[c] for c in CANONICAL]], 'native_bipolar'
    elif all(f'{a}-CS2' in lut and f'{b}-CS2' in lut for a, b in pairs):
        out = np.empty((N_CHANNELS, D.shape[1]))
        for k, (a, b) in enumerate(pairs):
            out[k] = D[lut[f'{a}-CS2']] - D[lut[f'{b}-CS2']]
        mode = 'rederived_cs2'
    elif all(a in lut and b in lut for a, b in pairs):
        out = np.empty((N_CHANNELS, D.shape[1]))
        for k, (a, b) in enumerate(pairs):
            out[k] = D[lut[a]] - D[lut[b]]
        mode = 'rederived_bare'
    else:
        raise ValueError(f'{path}: no derivation path')

    out = out * 1e6                             # volts -> microvolts
    p99 = np.percentile(np.abs(out), 99)
    if not 1.0 < p99 < 1e4:
        raise ValueError(f'{path} [{mode}]: p99 {p99:.1f} uV is not scalp EEG')
    return out, mode


# --- 6. causal filter -------------------------------------------------------
class CausalBandpass:
    """Forward-only Butterworth. Stream equivalence is bit-identical (0.000e+00).
    order=2 yields a realised 4th-order bandpass (2 SOS sections).
    Never filtfilt — non-causal, leaks future samples."""

    def __init__(self, sfreq=256.0, l_freq=0.5, h_freq=40.0, order=2, n_channels=18):
        nyq = sfreq / 2.0
        self.sfreq, self.n_channels = float(sfreq), int(n_channels)
        self.sos = signal.butter(order, [l_freq/nyq, h_freq/nyq],
                                 btype='band', output='sos')
        self._zi_proto = signal.sosfilt_zi(self.sos)     # (n_sections, 2)
        self.reset()

    def reset(self, primer=None):
        """Call PER RECORD. Priming gives 27x less startup transient."""
        zi = np.repeat(self._zi_proto[:, None, :], self.n_channels, axis=1)
        self._zi = zi * (np.asarray(primer, float)[None, :, None]
                         if primer is not None else 0.0)

    def process(self, block):
        block = np.asarray(block, dtype=np.float64)
        out, self._zi = signal.sosfilt(self.sos, block, axis=-1, zi=self._zi)
        return out

    def group_delay_ms(self, freqs):
        b, a = signal.sos2tf(self.sos)
        _, gd = signal.group_delay((b, a), w=np.asarray(freqs, float), fs=self.sfreq)
        return gd / self.sfreq * 1e3


# --- 7. windowing + labelling ----------------------------------------------
LAB_ICTAL, LAB_INTER, LAB_DROP = 1, 0, -1
WINDOW_SEC, STRIDE_SEC, GUARD_SEC = 2.0, 1.0, 60.0


def label_windows(starts, window_sec, seizures, guard_sec):
    """Ictal is applied BEFORE the guard, so ictal wins. Guard windows are
    DROPPED, not called interictal."""
    ends = starts + window_sec
    lab = np.full(starts.shape, LAB_INTER, dtype=np.int8)
    ev  = np.full(starts.shape, -1, dtype=np.int16)
    for k, (s0, s1) in enumerate(seizures):
        ov = (starts < s1) & (ends > s0)
        lab[ov] = LAB_ICTAL; ev[ov] = k
    for s0, s1 in seizures:
        near = (starts < s1 + guard_sec) & (ends > s0 - guard_sec)
        lab[near & (lab != LAB_ICTAL)] = LAB_DROP
    return lab, ev


def window_record(path, summary, sfreq=256.0, window_sec=WINDOW_SEC,
                  stride_sec=STRIDE_SEC, guard_sec=GUARD_SEC):
    record = os.path.basename(path)
    if summary is None:
        if not summarized:
            raise RuntimeError('`summarized` is empty — rebuild it')
        if record not in KNOWN_UNANNOTATED:
            raise ValueError(f'{record}: no summary entry, not a known unannotated record')
        return None

    X, derivation = load_18(path)
    bp = CausalBandpass(sfreq=sfreq, n_channels=N_CHANNELS)
    bp.reset(primer=X[:, 0])                    # per record, never across records
    X = bp.process(X)

    w_size = int(round(window_sec * sfreq))     # duration, not sample count
    w_stride = int(round(stride_sec * sfreq))
    if X.shape[1] < w_size:
        return None
    W = sliding_window_view(X, w_size, axis=-1)[:, ::w_stride, :].transpose(1, 0, 2)

    starts = np.arange(W.shape[0], dtype=np.float64) * stride_sec
    lab, ev = label_windows(starts, window_sec, summary.seizures, guard_sec)
    keep = lab != LAB_DROP
    if not keep.any():
        return None

    meta = {'y': lab[keep], 't_start': starts[keep], 'record': record,
            'subject': record.split('_')[0],
            'subject_group': subject_group_of(record),
            'seizure_event': [f'{record}#{e}' if e >= 0 else '' for e in ev[keep]],
            'derivation': derivation, 'noisy': record in NOISY_RECORDS}
    return W[keep], lab[keep], meta


# --- 8. feature extractor ---------------------------------------------------
BANDS = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,13),
         'beta': (13,30), 'gamma': (30,40)}
_EPS = 1e-12          # safe ONLY because load_18 returns microvolts
FAMILIES = ('hjorth','linelength','dwt','bandpower','spec_entropy','aperiodic')


class FeatureExtractor:
    """All constants precomputed at construction; the hot path allocates only
    its output and never loops over channels in Python."""

    def __init__(self, sfreq=256.0, n_channels=18, window_size=512,
                 families=FAMILIES, fit_lo=1.0, fit_hi=40.0,
                 wavelet='db4', dwt_level=4):
        self.sfreq, self.n_channels, self.window_size = sfreq, n_channels, window_size
        self.families, self.wavelet, self.dwt_level = tuple(families), wavelet, dwt_level

        self._taper = np.hanning(window_size)
        self._tnorm = float((self._taper**2).sum())
        self._freqs = np.fft.rfftfreq(window_size, 1/sfreq)
        self._masks = {n: (self._freqs >= lo) & (self._freqs < hi)
                       for n, (lo, hi) in BANDS.items()}
        self._support = (self._freqs >= 1.0) & (self._freqs < 40.0)
        self._log2n = float(np.log2(max(int(self._support.sum()), 2)))

        # pinv depends only on the frequency grid -> CONSTANT. This is what makes
        # the aperiodic exponent real-time viable (0.002 ms vs 0.623 ms polyfit).
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
            s += ['hjorth_activity','hjorth_mobility','hjorth_complexity']
        if 'linelength' in self.families:   s += ['linelength']
        if 'dwt' in self.families:
            s += [f'dwt_cD{i}' for i in range(1, self.dwt_level+1)]
            s += [f'dwt_cA{self.dwt_level}']
        if 'bandpower' in self.families:    s += [f'relpow_{b}' for b in BANDS]
        if 'spec_entropy' in self.families: s += ['spec_entropy']
        if 'aperiodic' in self.families:    s += ['aperiodic_slope']
        return s

    def _names(self):
        chans = (list(CANONICAL) if self.n_channels == N_CHANNELS
                 else [f'ch{i:02d}' for i in range(self.n_channels)])
        return ([f'{c}__{s}' for c in chans for s in self._suffixes()]
                + ['guard_frontal','guard_temporal','guard_global'])

    def transform_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2: X = X[None]
        blocks = []

        if any(f in self.families for f in ('bandpower','spec_entropy','aperiodic')):
            spec = np.fft.rfft(X * self._taper, axis=-1)
            psd = (spec.real**2 + spec.imag**2) / self._tnorm + _EPS

        if 'hjorth' in self.families:
            d1, d2 = np.diff(X, axis=-1), np.diff(X, n=2, axis=-1)
            v0, v1, v2 = X.var(-1)+_EPS, d1.var(-1)+_EPS, d2.var(-1)+_EPS
            mob = np.sqrt(v1/v0)
            blocks.append(np.stack([np.log10(v0), mob, np.sqrt(v2/v1)/mob], -1))

        if 'linelength' in self.families:
            ll = np.mean(np.abs(np.diff(X, axis=-1)), axis=-1)
            blocks.append(np.log10(ll + _EPS)[..., None])

        if 'dwt' in self.families:
            c = pywt.wavedec(X, self.wavelet, level=self.dwt_level, axis=-1)
            # [cA_n, cD_n, ..., cD1] -- REVERSE frequency order
            details = c[1:][::-1]
            e = [np.log10((d**2).mean(-1) + _EPS) for d in details]
            e.append(np.log10((c[0]**2).mean(-1) + _EPS))
            blocks.append(np.stack(e, -1))

        if 'bandpower' in self.families:
            tot = psd[..., self._support].sum(-1, keepdims=True) + _EPS
            blocks.append(np.stack([psd[..., m].sum(-1)/tot[...,0]
                                    for m in self._masks.values()], -1))

        if 'spec_entropy' in self.families:
            p = psd[..., self._support]
            p = p / (p.sum(-1, keepdims=True) + _EPS)
            blocks.append((-(p*np.log2(p+_EPS)).sum(-1) / self._log2n)[..., None])

        if 'aperiodic' in self.families:
            blocks.append((np.log10(psd[..., self._fit_mask]) @ self._pinv.T)[..., 1:2])

        flat = np.concatenate(blocks, -1).reshape(X.shape[0], -1)
        front = [CANONICAL.index(c) for c in ('FP1-F7','FP1-F3','FP2-F4','FP2-F8')]
        temp  = [CANONICAL.index(c) for c in ('F7-T7','T7-P7','F8-T8','T8-P8')]
        guards = np.stack([
            np.log10(X[:, front].var(axis=(1,2)) + _EPS),
            np.log10(np.diff(X[:, temp], axis=-1).var(axis=(1,2)) + _EPS),
            np.log10(X.var(-1).mean(-1) + _EPS)], -1)
        return np.concatenate([flat, guards], -1).astype(np.float32)


fx = FeatureExtractor()


# --- 9. evaluation ----------------------------------------------------------
def window_metrics(y_true, scores):
    """AUPRC is meaningless without its baseline. At a 0.339% positive rate a
    useless model scores ~0.0034; read auprc_lift."""
    y_true = np.asarray(y_true)
    pos_rate = float(y_true.mean())
    out = {'positive_rate': pos_rate, 'auprc_baseline': pos_rate}
    if len(np.unique(y_true)) < 2:
        return {**out, 'auroc': np.nan, 'auprc': np.nan, 'auprc_lift': np.nan}
    out['auroc'] = float(roc_auc_score(y_true, scores))
    out['auprc'] = float(average_precision_score(y_true, scores))
    out['auprc_lift'] = out['auprc'] / pos_rate
    return out


def _persistence(alarm, k, n):
    if k <= 1 and n <= 1: return alarm
    c = np.convolve(alarm.astype(int), np.ones(n, int), mode='full')[:len(alarm)]
    return c >= k


def event_metrics(meta, scores, threshold, window_sec=2.0, stride_sec=1.0,
                  k_of_n=(3,5), refractory_sec=30.0, tolerance_sec=30.0,
                  subsample_rate=1.0):
    """Event-level sensitivity, FA/h, latency.

    subsample_rate: if interictal windows were kept at rate r, each survivor
    represents 1/r seconds, so the FA/h denominator divides by r.

    CAVEAT: refractory merging collapses continuous alarming into one event, so
    an always-on model scores a deceptively low FA/h. Always read alarm_duty.
    """
    meta = meta.reset_index(drop=True).copy()
    meta['score'] = np.asarray(scores)
    k, n = k_of_n
    detected, latencies = [], []
    n_fa = n_inter = n_alarm_inter = 0

    for _, grp in meta.groupby('record', observed=True, sort=False):
        grp = grp.sort_values('t_start')
        t, y = grp.t_start.to_numpy(), grp.y.to_numpy()
        ev = grp.seizure_event.to_numpy()
        alarm = _persistence(grp.score.to_numpy() >= threshold, k, n)
        t_dec = t + window_sec                  # decision available at window END

        for eid in pd.unique(ev[y == 1]):
            m = ev == eid
            onset = float(t[m].min()); offset = float(t[m].max()) + window_sec
            hit = alarm & (t_dec >= onset) & (t_dec <= offset + tolerance_sec)
            detected.append(bool(hit.any()))
            if hit.any():
                latencies.append(float(t_dec[hit].min() - onset))

        inter = y == 0
        n_inter += int(inter.sum())
        fa = alarm & inter
        n_alarm_inter += int(fa.sum())
        if fa.any():
            ft = t_dec[fa]
            n_fa += 1 + int((np.diff(ft) > refractory_sec).sum())

    inter_hours = n_inter * stride_sec / 3600.0 / subsample_rate
    return {'threshold': float(threshold), 'n_events': len(detected),
            'sensitivity_event': float(np.mean(detected)) if detected else np.nan,
            'false_alarms': n_fa, 'interictal_hours': inter_hours,
            'fa_per_hour': n_fa/inter_hours if inter_hours > 0 else np.nan,
            'alarm_duty': n_alarm_inter/n_inter if n_inter else np.nan,
            'latency_median_s': float(np.median(latencies)) if latencies else np.nan,
            'latency_p90_s': float(np.percentile(latencies, 90)) if latencies else np.nan}


def report_folds(per_fold, metrics=('auprc','auprc_lift','auroc','sensitivity_event')):
    """Per-fold rows, spread, and the worst fold by name. NEVER the mean alone —
    44% of positives come from four subjects and a mean hides everything."""
    df = pd.DataFrame(per_fold)
    print(df.to_string(index=False)); print()
    for mn in metrics:
        if mn not in df: continue
        v = df[mn].dropna()
        if not len(v): continue
        worst = df.loc[v.idxmin()]
        print(f'{mn:18s} median {v.median():7.4f}  mean {v.mean():7.4f}  '
              f'min {v.min():7.4f}  max {v.max():7.4f}  '
              f'(worst: {worst.get("held_out","?")})')
    return df


# --- 10. LOSO ---------------------------------------------------------------
META_COLS = ('y','subject_group','subject','record','seizure_event','t_start',
             'derivation','noisy')


def load_subjects(groups, subsample=1.0, seed=0, features=FEATURES):
    """Keeps ALL positives; samples negatives at `subsample`.

    Pass subsample=1.0 for TEST folds — persistence smoothing and detection
    latency both require contiguous windows.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for f in sorted(glob.glob(f'{features}/*.parquet')):
        d = pd.read_parquet(f)
        d = d[d.subject_group.isin(groups)]
        if not len(d):
            del d; continue
        if subsample < 1.0:
            keep = (d.y == 1).to_numpy() | (rng.random(len(d)) < subsample)
            d = d[keep]
        frames.append(d); del d
    out = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    return out


def split_xy(df):
    feat = [c for c in df.columns if c not in META_COLS]
    return (df[feat].to_numpy(dtype=np.float32),
            df.y.to_numpy(dtype=np.int8),
            df[list(META_COLS)].copy())


def make_model(name, seed=0):
    """Scaler lives INSIDE the pipeline — fitting it before the split leaks."""
    if name == 'logreg':
        return Pipeline([('scale', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=2000,
                                                    class_weight='balanced', C=1.0))])
    if name == 'svm_linear':
        return Pipeline([('scale', StandardScaler()),
                         ('clf', LinearSVC(C=0.1, class_weight='balanced',
                                           dual='auto', max_iter=5000))])
    if name == 'rf':
        return Pipeline([('clf', RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced_subsample', n_jobs=-1, random_state=seed))])
    raise ValueError(name)


def scores_of(model, X):
    if hasattr(model[-1], 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_loso(models=('logreg',), subsample=0.1, seed=0, features=FEATURES,
             groups=None, save_partial=f'{OUT}/loso_folds_partial.csv'):
    """LOSO with per-fold incremental save, so a disconnect costs nothing."""
    if groups is None:
        groups = sorted({str(g) for f in glob.glob(f'{features}/*.parquet')
                         for g in pd.read_parquet(f, columns=['subject_group'])
                         .subject_group.unique()})
    print(f'{len(groups)} groups: {groups}\n')

    rows = []
    for held in groups:
        te = load_subjects([held], subsample=1.0, features=features)
        if (te.y == 1).sum() == 0:
            print(f'{held}: no seizures held out, skipping'); del te; continue
        Xte, yte, mte = split_xy(te); del te

        tr = load_subjects([g for g in groups if g != held],
                           subsample=subsample, seed=seed, features=features)
        Xtr, ytr, _ = split_xy(tr); del tr; gc.collect()

        for name in models:
            t0 = time.perf_counter()
            model = make_model(name, seed); model.fit(Xtr, ytr)
            s = scores_of(model, Xte)
            rec = {'model': name, 'held_out': held,
                   'n_train': len(ytr), 'n_test': len(yte),
                   'test_events': mte.loc[yte == 1, 'seizure_event'].nunique()}
            rec.update(window_metrics(yte, s))
            rec.update(event_metrics(mte, s, np.quantile(s, 0.995),
                                     subsample_rate=1.0))
            rec['fit_s'] = time.perf_counter() - t0
            rows.append(rec)
            if save_partial:
                pd.DataFrame(rows).to_csv(save_partial, index=False)
            print(f'  {name:11s} {held:7s} AUPRC {rec["auprc"]:.4f} '
                  f'(lift {rec["auprc_lift"]:5.1f}x)  sens {rec["sensitivity_event"]:.2f} '
                  f'@ {rec["fa_per_hour"]:5.2f} FA/h  duty {rec["alarm_duty"]:.3f}  '
                  f'lat {rec["latency_median_s"]:.1f}s  [{rec["fit_s"]:.0f}s]')

        del Xtr, ytr, Xte, yte, mte; gc.collect()
    return pd.DataFrame(rows)


# --- 11. verify -------------------------------------------------------------
assert norm('T8-P8-0') == 'T8-P8' and norm('FT9-FT10') == 'FT9-FT10'
assert norm('01') == 'O1'
assert subject_group_of('chb21_19.edf') == 'chb01'
assert subject_group_of('chb17a_03.edf') == 'chb17'
assert fx.n_features == 291, fx.n_features
_l, _ = label_windows(np.arange(0, 200, 1.0), 2.0, [(50., 55.), (70., 75.)], 20.)
assert _l[np.arange(0, 200, 1.0) == 69.0][0] == LAB_ICTAL, 'ictal must beat guard'

print(f'\nrestored | {len(summarized)} summaries | fx {fx.n_features} features | '
      f'{n_shard} shards')
print('symbols: norm subject_group_of parse_summary summarized load_18 '
      'CausalBandpass window_record fx window_metrics event_metrics '
      'report_folds load_subjects split_xy make_model run_loso')

# ============================================================================
# IF THE VM WAS RECYCLED (n_edf == 0):
#   !pip install -q awscli mne PyWavelets pyarrow
#   !aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ /content/chbmit/
#
# IF SHARDS ARE MISSING (n_shard < 26):
#   from google.colab import drive; drive.mount('/content/drive')
#   for f in sorted(glob.glob(f'{DRIVE}/features/*.parquet')):
#       shutil.copy(f, FEATURES)
# ============================================================================
