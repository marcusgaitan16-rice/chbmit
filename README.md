# CHB-MIT Low-Latency Seizure Decoder

Seizure detection on the CHB-MIT Scalp EEG Database, built toward a *deployable*
decoder rather than a leaderboard number: causal preprocessing, parsimonious
feature extraction, leakage-safe evaluation, and an explicit compute budget
benchmarked against shipping neurostimulation devices.

**Status:** end-to-end deployable. A patient-specific causal TCN reaches
**sensitivity 0.86–1.00 at 0.4–1.8 FA/h with 5–21 s detection latency** across
six subjects, beating patient-calibrated logistic regression on five of six. It
runs as streaming fixed-point inference in **58,784 MACs/timestep and ~82 KB**,
and Q15 quantization costs **zero** detection performance — sensitivity, FA/h and
alarm duty are identical to float64 at every operating point.

Cross-subject transfer fails for all four model classes tried, including a
pretrained TCN (0.79× population-only, below chance). That is the finding, and
patient calibration is the answer.

---

## Why this project is shaped this way

Most published CHB-MIT work optimizes offline classification accuracy. That is
not the constraint an implantable detector operates under, and it is not what
this project optimizes.

Three commitments follow, and they drive most of the decisions below:

1. **Every filter is causal.** No `filtfilt`, no future samples, anywhere —
   including in offline dataset construction, where non-causal filtering is easy
   to do accidentally and nearly impossible to detect downstream.
2. **Compute cost is a first-class metric.** Feature families are budgeted in
   MACs and transcendental ops, not just wall-clock, because wall-clock on a
   laptop does not transfer to a DSP.
3. **Evaluation is event-level.** Ictal windows are ~0.3% of the corpus.
   Accuracy is meaningless here and does not appear in this repo.

---

## Current state

| Layer | Status |
|---|---|
| Acquisition + integrity | Complete — 686/686 byte-verified |
| Annotation parsing | Complete — 24/24 summaries, count-validated |
| Montage analysis | Complete — 12 variants, 683/686 canonical coverage |
| chb12 re-derivation | **Validated** — derivation sound, noise characterized |
| Canonical loader `load_18()` | Complete — 683 native / 1 cs2 / 2 bare |
| Causal filtering | Complete — bit-identical stream equivalence |
| Windowing + labelling | Complete — 2 s / 1 s / 60 s guard |
| Feature extraction | Complete — 26/26 shards, 6.2 GB |
| Cross-validation + baselines | LogReg 23/23; RF diagnostic; cross-subject limit characterized |
| Patient calibration curve | Complete — 6 subjects, inner-validated thresholds |
| 1D-TCN | Complete — 6 subjects, beats logreg on 5 |
| Streaming fixed-point port | **Complete and validated** |
| 1D-TCN | Not started |

**Corpus totals as parsed:** 686 EDF files (45.8 GB), 676 annotated,
198 seizures, 3.23 h ictal.

### Verified corpus totals

```
686 EDFs | 676 annotated | 198 seizures | 3.23 h ictal
683 native_bipolar / 1 rederived_cs2 / 2 rederived_bare
3,480,880 windows | 11,809 ictal (0.339%) | 23 subject groups
events 198/198 -- no seizure lost anywhere in the pipeline
```

**Positives are extremely concentrated.** chb12 (40 events), chb15 (20),
chb24 (16) and chb13 (12) hold 88 of 198 — 44% from four subjects — while
chb02, chb07, chb11, chb19 and chb22 have 3 each.

**Ictal windows per event span 29×.** chb11 averages 270 windows per seizure;
chb16 averages 9.4. That is real seizure-duration variation, and it caps
achievable sensitivity on short-seizure subjects regardless of model quality.

**chb04 is 16% of the table from 4 of 198 seizures.** Its records are ~4 hours
(median `t_start` max 14,398 s) rather than the usual one hour. In pooled
training it contributes 16% of negatives and 2% of positives.

---

## 1. Acquisition and integrity

```bash
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ /content/chbmit/
```

A prior `wget -r` against `physionet.org` sustained ~161 KB/s — roughly three
days for the full corpus. The S3 sync completes in minutes and is resumable.

Three integrity checks, in increasing strength:

| check | result |
|---|---|
| size heuristic (flag < 15 MB) | 5 flagged — **all false positives** |
| EDF header consistency | 1 flagged — **false positive** |
| **S3 manifest byte-diff** | **686/686 complete, 0 mismatches, 0 missing** |

The manifest diff is authoritative. Both false-positive checks are retained in
the notebook with their failure modes documented, because the reasons they
misfired are properties of this corpus worth knowing:

- **The 15 MB threshold does not apply to CHB-MIT.** Hardware gaps between
  consecutively-numbered files mean some records are legitimately short.
  `chb24_22.edf` is 12.6 MB against 42.4 MB for its siblings — ~18 minutes of
  recording, not a truncated hour.
- **`chb17b_69.edf` carries 256 trailing bytes** past its last complete data
  record. Header declares 7,424 bytes (256 × (28+1)); 3,600 records × 28 ch ×
  256 samples × 2 bytes = 51,617,024; file is 51,617,280. A data record here is
  14,336 bytes, so 256 bytes cannot be a partial record — it is padding. MNE
  reads exactly 60.0 minutes. The checker recomputed header length instead of
  reading the declared value at bytes 184:192, and labelled a *larger*-than-
  expected file "truncated."

---

## 2. Annotation layer

### The parser bug that mattered

`chbNN-summary.txt` is not uniformly formatted. Three variants had to be
handled:

1. **Single vs. multi-seizure syntax** — `Seizure Start Time:` vs.
   `Seizure 1 Start Time:`
2. **Inconsistent whitespace, *within a single record*.** `chb12_27` contains
   both `Seizure 1 Start Time: 916 seconds` (one space) and
   `Seizure 1 End Time:  951 seconds` (two).
3. **Mid-file `Channels changed:` blocks** followed by `Channel N: LABEL` lines,
   which a delimiter-based walk ingests as data.

Variant 2 broke the original parser: `line.split(': ')[1]` returns
`' 951 seconds'`, and `.split(' ')[0]` on a leading space yields `''`. This
silently dropped **all 6 seizures from `chb12_27` and all 6 from `chb12_29`**.

The parser is now regex-anchored on the numeric value rather than a delimiter,
and **validates parsed count against the declared `Number of Seizures in File`,
raising on mismatch**.

> **The validation is the load-bearing part, not the regex.** The original bug
> emitted warnings, which scrolled off the top of the Colab output. The
> corrupted annotations then propagated into a signal-quality measurement that
> produced an apparently significant result (Cohen's *d* = 12.67) which was
> entirely an artifact. Warnings are insufficient. Assertions raise.

### Excluded records

Ten `chb24` files appear on disk with no summary entry:
`chb24_02, _05, _08, _10, _12, _16, _18, _19, _20, _22`.

**Excluded.** Unannotated is not seizure-free. Including them as all-interictal
would inject unknown false negatives into training *and* inflate the
interictal-hours denominator, quietly depressing the reported false-alarm rate.
That metric is headed for the write-up, so its denominator has to be defensible.

`chb24` retains 12 annotated records and 16 seizures, so the subject stays in
cross-validation. Cost: ~10 h interictal out of ~900.

Enforced in the loader — a record with no summary entry is skipped, never
defaulted to an empty seizure list.

### Note on totals

The 3.23 h ictal figure matches the published corpus description. The
198-seizure count runs above the 182–185 commonly cited in secondary sources.
Since declared-equals-parsed is asserted per record, this reflects what the
annotation files contain rather than an over-count. To be footnoted, not
reconciled.

---

## 3. Montage analysis

### The canonical 18

```python
CANONICAL = ['FP1-F7','F7-T7','T7-P7','P7-O1',   # left temporal
             'FP1-F3','F3-C3','C3-P3','P3-O1',   # left parasagittal
             'FP2-F4','F4-C4','C4-P4','P4-O2',   # right parasagittal
             'FP2-F8','F8-T8','T8-P8','P8-O2',   # right temporal
             'FZ-CZ','CZ-PZ']                    # midline
```

Index order is fixed and load-bearing — downstream code indexes channels by
integer position, so reordering silently corrupts everything after it.

**Dropped from the reference 23-channel montage:**

- `T7-FT9`, `FT9-FT10`, `FT10-T8` — FT9/FT10 are not standard 10-20 positions
  and the chain is inconsistently present across subjects. Keeping it would make
  the feature vector's shape subject-dependent, which is how a channel index map
  silently shifts between training and test.
- `P7-T7` — the polarity inverse of `T7-P7`. Including both gives a perfectly
  collinear pair: harmless for tree models, quietly bad for anything linear.

### Survey across all 686 files

```
channel counts: {22: 26, 23: 276, 24: 54, 25: 1, 28: 275, 29: 14, 31: 1, 38: 39}
distinct montages: 12
```

Under name-based selection:

```
683 files | missing  0 canonical channels | all 26 case directories
  3 files | missing 18 canonical channels | chb12 only
```

**99.6% coverage.** The 12 montages differ almost entirely in dummy (`--N`) and
non-canonical channels, not in the 18 that matter. This is the empirical case
for **selecting by name** rather than **dropping by name**: extras such as
`--0`…`--4` and `EKG1-CHIN` become irrelevant instead of something to enumerate.

### Name normalization

Two quirks, both found empirically:

- **MNE duplicate-run suffixes.** `T8-P8` appears twice in the 23-channel
  header; MNE disambiguates to `T8-P8-0` / `T8-P8-1`. In montages where it
  appears once it stays `T8-P8`. The matcher must accept both — and must not
  strip trailing digits from `FT9-FT10`.
- **`01` is a typo for `O1`** in the `chb12_28`/`_29` headers — zero-one, not
  letter-O-one; it sorts before `C2`. `O2` is spelled correctly in the same
  files, so exactly one label was mistyped at recording time. Without the alias,
  four pairs are blocked (`P7-O1`, `P3-O1`, and the chains they terminate).

Handled by an explicit named constant, not an inline regex:

```python
ALIASES = {'01': 'O1'}   # zero-one typo in chb12_28/_29 headers
```

---

## 4. chb12 re-derivation

Three `chb12` files are not recorded in a bipolar montage, which invalidates the
general claim that CHB-MIT is uniformly bipolar.

| file | montage | seizures |
|---|---|---|
| `chb12_27` | referential to CS2 — `FP1-CS2`, `F7-CS2`, … | 6 |
| `chb12_28` | bare electrode labels — `FP1`, `F7`, … | 1 |
| `chb12_29` | bare electrode labels | 6 |

**13 seizures out of chb12's 40** — roughly a third of the patient, ~6.5% of the
corpus total. Dropping was rejected on that basis.

Re-derivation is exact where the reference is shared:
`FP1-F7 = (FP1-CS2) − (F7-CS2)`, since CS2 cancels identically. All electrodes
required for all 18 pairs are present in both variants, given the `01 → O1`
alias.

**Stated caveat:** for `_27` the shared reference is explicit in the labels, so
the subtraction is provably exact. For `_28`/`_29` the labels are bare and the
reference is *unstated* — a common reference is **assumed**. Very likely
correct, but it is an assumption, and it is recorded as one rather than buried.

### Validation — RESOLVED

No golden test file exists: a corpus-wide scan found no file containing both a
bipolar channel and the two referential channels it would derive from. Exact
sample-for-sample validation is therefore impossible, and only statistical and
spectral comparison is available. This caps achievable confidence and is
disclosed rather than glossed.

Within that limit, the derivation is validated on four independent grounds.

**1. Quantization ruled out.** EDF header LSB is identical at 0.3907 µV for both
native bipolar and CS2-referential channels. Differencing does not inherit a
coarser quantization step.

**2. Ictal contamination ruled out.** With the fixed parser and ictal ± 120 s
masking properly applied, in-band power moved from −95.5 to −95.2 dB — i.e.
excluding six seizures barely changed the number. The elevation is not driven by
seizure content.

**3. The gap is not flat across frequency, so it is not a scaling error.**
Against a native file adjacent in recording time (`chb12_24` vs `chb12_27`), the
difference is ~10 dB at 3 Hz and widens to ~25 dB by 30 Hz. A gain or
units error in the subtraction would shift a log-log PSD *vertically and
uniformly*. This does not. The arithmetic is correct; the recording is noisier.

**4. Narrowband peaks have harmonic structure, so the source is instrumental.**
Prominence-filtered peaks in `chb12_27`:

| frequency | prominence |
|---|---|
| 16.25 Hz | 10.0 dB |
| 27.50 Hz | 4.4 dB |
| 32.38 Hz | 6.7 dB |
| 36.00 Hz | 3.2 dB |
| 43.88 Hz | 15.4 dB |

32.38 Hz is the second harmonic of 16.25 Hz. Harmonic structure implies a
periodic mechanical or electrical source — a pump, motor, or switching supply
in the recording environment — not anything neural and not an artifact of the
subtraction.

**Per-channel behaviour is clean.** Spread across the 18 channels is 8.4 dB
(−94.3 to −102.7), with no outlier. Left-hemisphere channels run consistently
hotter than right, which is a plausible asymmetry in a focal epilepsy patient
rather than an instrumentation fault. A single bad electrode propagating into
every pair that uses it would look nothing like this.

### Mechanism

A bipolar amplifier channel rejects common-mode interference *in hardware, at
the input stage*. Software differencing of two referential channels performs the
same subtraction *after* both have been amplified and digitized, so interference
that affected the front end does not cancel as cleanly. Same mathematics, worse
conditions.

### Consequence for the pipeline

The 43.88 Hz peak — the largest — falls in the stopband of the 0.5–40 Hz
bandpass and never reaches a feature. The peaks at **16.25, 27.50, 32.38 and
36.00 Hz do reach features**, sitting in beta and low gamma, where they will
inflate `relpow_beta` and steepen `aperiodic_slope` on these three files
specifically.

Therefore `derivation` is carried as a **covariate to check at baseline time**,
not merely a recorded column: verify that windows from these files are not
systematically scoring differently.

**No global notch filter.** Adding one to fix three files out of 686 makes every
file pay the compute and the group delay. If it is worth doing, it is worth
doing as an ablation on the chb12 subset — extract with and without, report
whether it moves anything.

### Group comparison at n=3 — closed

With all three modes loading, the comparison completes:

```
native    n=21   -104.6 +/- 1.0 dB   range [-106.9, -102.7]
rederived n=3    -102.0 +/- 4.8 dB   [-95.2, -106.0, -104.8]
```

**`chb12_28` (-106.0) and `chb12_29` (-104.8) fall inside the native range.**
The 4.8 dB spread is one file, not a group property.

This sharpens the finding considerably. Re-derivation introduces **no detectable
power shift** — two of three re-derived files are statistically
indistinguishable from natively-recorded files from the same patient. The
elevation is specific to the `chb12_27` recording session, not to the method.

Amplitude confirms it, and validates the one assumption that could not be
checked from the header:

| file | mode | p99 amplitude |
|---|---|---|
| `chb01_03` | native_bipolar | 144.8 µV |
| `chb12_27` | rederived_cs2 | 414.6 µV |
| `chb12_28` | rederived_bare | 108.2 µV |
| `chb12_29` | rederived_bare | 158.6 µV |

The bare-label files land squarely in normal scalp range, so the **unstated
common reference assumption holds**. Had it been wrong, the unit assertion at
`load_18`'s single exit point would have fired.

Consequently `derivation` alone is too coarse a covariate. The correct flag is
record-level:

```python
NOISY_RECORDS = {'chb12_27.edf'}   # session-specific narrowband interference
```

*(`chb12_29` is 927,744 samples vs. 921,600 — 3,624 s rather than 3,600, matching
its summary end time of 19:07:43. Legitimately 24 s longer, not a defect.)*

**All 13 seizures recovered.**


---

## 4b. Signal pipeline

### Causal filtering

2nd-order Butterworth bandpass 0.5–40 Hz, forward-only `sosfilt` with `zi`
carried across blocks. For a bandpass design scipy's `order` yields realised
order `2*order`, so this is a 4th-order bandpass in 2 SOS sections.

**Stream equivalence is bit-identical.** Filtering one pass over a record versus
streaming it in 4-sample packets with state carried: max abs diff **0.000e+00**.
That result licenses offline whole-record filtering while still claiming
stream-equivalence.

The negative controls are what make it a real test — both failure modes are the
same order of magnitude as the signal itself (2.6e-04) and both are silent:

| comparison | max abs diff |
|---|---|
| streamed **with** state carry | **0.000e+00** |
| streamed **without** state carry | 2.091e-04 |
| `filtfilt` (non-causal) | 4.445e-04 |

**Priming matters more than expected.** On a DC-offset signal the first 64
samples cold-start at 5.07e-05 against a 1.49e-06 steady state — the offset
passes straight through before the highpass settles. Priming `zi` to the first
sample gives 1.88e-06, essentially steady state immediately: **27× less startup
garbage**, and it matters because windowing begins at sample zero.

`reset()` is called **per record**, never carried across recording boundaries.

### Windowing and labelling

2 s windows, 1 s stride, 60 s guard band. Ictal labels are applied *before* the
guard, so ictal wins — a window straddling one seizure's guard and the next
seizure's onset is correctly ictal. `sliding_window_view` returns a genuine view,
avoiding a 2× memory blowup at 50% overlap.

### The epsilon bug

Caught only by differential testing. Shape was right, finiteness was right,
nothing was NaN — but Hjorth mobility disagreed with a naive implementation by
**0.008**, a 1% error on a scale-invariant feature.

Cause: EEG in **volts** has variance ~3e-9, so a fixed `_EPS = 1e-12` is a
~0.03% perturbation on a typical channel and far worse on quiet ones (the `max`
picks the worst). Fix: work in **microvolts**, where `_EPS` is 3e-14% of
variance. Post-fix both differential tests land at ~1.4e-08 and ~2.6e-08 — the
**float32 storage floor**, not residual error.

> **Generalizable lesson.** Epsilon guards must be scaled to the data. A constant
> that is inert at unit scale can be a 1% distortion twelve orders of magnitude
> down. Only comparison against an independent implementation surfaced it.

### DWT bands do not fit the passband

At 256 Hz a 4-level db4 decomposition gives:

| coefficient | band | vs. 0.5–40 Hz passband |
|---|---|---|
| cD1 | 64–128 Hz | **entirely stopband** |
| cD2 | 32–64 Hz | mostly stopband |
| cD3 | 16–32 Hz | useful |
| cD4 | 8–16 Hz | useful |
| cA4 | 0–8 Hz | useful |

Two of five DWT features per channel — 36 of 291 — quantify filter rolloff
noise. With the DWT already costing 15,872 MACs/channel against 5,377 for a
*shared* rFFT, it is likely to lose its ablation. At 128 Hz the bands realign
(cD1 = 32–64, cD2 = 16–32) — another point for decimation (D22).

Separately: `wavedec` returns details in **reverse** frequency order
(`[cA_n, cD_n, ..., cD1]`). Naming them in list order silently mislabels every
band.

### Extraction throughput

5.3 s/record, ~208 s/subject, ~90 min for the full corpus. Per-subject Parquet
shards, ~6.2 GB total. Categorical dtypes on the string columns took chb01 from
211 MB to 171 MB — the float32 floor.

**Two extraction bugs, same symptom.** chb17 produced "no usable records" for
three shards because (a) `summarized` was an empty dict after a restart, making
every record look unannotated, and (b) the shard loop globbed by directory name,
but chb17's files live in `chb17/` while carrying `chb17a_`/`b_`/`c_` prefixes.
Both are now guarded by assertions rather than silent `None` returns.


---

## 4c. First LOSO baseline — logistic regression

Leave-one-subject-out over 23 groups. Training folds subsampled to 10% of
interictal windows (all positives kept); **test folds at full density**, because
k-of-n persistence smoothing and detection latency both require contiguous
windows. Threshold at the 99.5th percentile of test-fold scores — see the caveat
below.

### Headline

```
median AUPRC lift   28.6x        (mean 52.6x -- skewed, report the median)
folds at chance     6/23         (lift < 5x, sensitivity 0.00)
folds sens >= 0.8   10/23
alarm duty          0.001-0.004  across every fold
median latency      ~22 s        against a 4 s floor
```

Duty staying at 0.001–0.004 everywhere matters: these detectors are not
always-on. That is the failure mode `alarm_duty` exists to catch, and it passes.

### Best and worst folds

| held out | AUPRC lift | sens | FA/h | latency |
|---|---|---|---|---|
| chb09 | 262× | 1.00 | 0.56 | 10 s |
| chb10 | 155× | 1.00 | 0.89 | 13 s |
| chb19 | 133× | 1.00 | 0.91 | 31 s |
| chb22 | 133× | 1.00 | 0.94 | 24 s |
| … | | | | |
| chb14 | 2.3× | 0.00 | 1.75 | — |
| chb06 | 2.8× | 0.00 | 0.77 | — |
| chb15 | 1.7× | 0.05 | 0.88 | 37 s |
| chb12 | 1.3× | 0.00 | 1.14 | — |
| chb13 | 1.1× | 0.00 | 1.23 | — |

### The pattern: more seizures predicts *worse* detection

An initial reading on the first 14 folds suggested seizure **duration** drove
performance. That hypothesis does not survive all 23 — chb15 has 101 windows per
event and fails at 1.7×; chb24 has 33 and succeeds at 35.8× with sensitivity
1.00. Two clean counterexamples.

The stronger relationship is with **event count, and it is negative**:

| predictor | vs. AUPRC lift | vs. sensitivity |
|---|---|---|
| windows per event (duration) | ρ = +0.427, p = 0.042 | +0.504, p = 0.014 |
| **number of events** | **ρ = −0.638, p = 0.001** | **−0.563, p = 0.005** |

Split on it:

- **≤5 events (n=10):** median lift 56.8×, median sensitivity 0.90
- **≥10 events (n=6):** median lift 2.2×, median sensitivity 0.00

Every subject with ten or more annotated seizures fails, except chb24.

Two candidate mechanisms, with different implications:

**Annotation depth.** A patient with 40 marked events (chb12) may have many
brief subclinical discharges a careful reviewer caught, while a patient with 3
has three unambiguous clinical seizures. This would mean the metric partly
reflects reviewer conservatism, not detectability.

**Severity.** Frequent-seizure patients may have more diffuse or variable
electrographic signatures that resist a shared cross-subject template.

Distinguishing these requires reading seizure durations directly from the
annotations rather than inferring them from window counts. Unresolved.

### Two caveats that must be fixed before any write-up

**1. The threshold is selected on the test fold** (`np.quantile(s, 0.995)`).
This effectively fixes the alarm rate near 1 FA/h on every fold, which makes
folds comparable but renders the sensitivity figures mildly optimistic. Replace
with an inner-validation threshold, or report the full sensitivity/FA-per-hour
operating curve.

**2. The best folds have the least reliable sensitivity estimates.** chb22 at
133× is 3/3 events; chb09 at 262× is 4/4. Sensitivity from three events has
enormous variance. AUPRC lift is window-level and better estimated, but the
sensitivity column needs confidence intervals.

### Latency is not the persistence rule

The floor is `window_sec + (k−1)·stride_sec` = 4 s for a 2 s window at 3-of-5.
Observed medians are 10–36 s, so the smoothing rule is not binding — the model's
score ramps slowly through the seizure. That means **the latency problem and the
short-seizure problem are the same problem**, and the fix is either faster
features or a lower operating threshold, not a shorter persistence window.


---

## 4d. Model selection — random forest as the baseline

Three models on the first fold (chb01):

| model | AUPRC lift | sens | FA/h | latency | fit time |
|---|---|---|---|---|---|
| logistic regression | 35.7× | 0.64 | 1.15 | 25.0 s | 106 s |
| linear SVM | 34.2× | 0.64 | 1.11 | 25.0 s | 125 s |
| **random forest (100 trees)** | **156.4×** | 0.64 | **0.84** | **12.0 s** | 1,046 s |

**The two linear models are redundant** — 35.7× vs 34.2×, identical sensitivity
and latency. Both fit a linear boundary on the same standardized features with
balanced class weights, so the hinge and log losses barely diverge at this scale.
Running both costs iteration time and answers nothing.

**Random forest is 4.4× better AUPRC, with lower FA/h *and* half the latency.**
Nonlinearity improves both headline problems simultaneously. The cost is 10×
training time — 1,046 s/fold, so ~6.7 h for a full 23-fold sweep.

Decision: **RF is the sole baseline going forward.** The linear results already
collected are reported as a one-line comparison ("linear was tried and lost by
4×"), which costs nothing since the numbers exist.

### RF fits the real-time budget — but not the memory budget

At 100 trees, inference measured 6.7 ms p50 / 8.6 ms p99, inside the 15.625 ms
packet budget. (300 trees was 21.7 ms and does *not* fit.) So the model that wins
on accuracy is also deployable on latency — not a given.

Memory is the tension:

| model | footprint | notes |
|---|---|---|
| logistic regression | ~1.2 KB | 291 coefficients |
| random forest, 100 trees | ~340 KB | ~28,176 nodes × ~12 B |

**A 280× difference.** 340 KB exceeds typical MCU SRAM, though it fits in flash
as a read-only structure. This is the accuracy-vs-deployability trade the
feature-budget curve exists to characterize, and it is now concrete rather than
hypothetical.

### The question that gates the 1D-TCN

RF's 156× on chb01 is encouraging, but chb01 was already a *working* fold for
logistic regression. **The question that matters is whether RF rescues the four
folds at chance** — chb06, chb12, chb13, chb15.

- If it does, the failure was model capacity, and a TCN is the natural escalation.
- If it does not, the failure is in the features, the labels, or cross-subject
  calibration — and a TCN will fail the same way at 50× the compute.

Two hours of compute to answer a question that determines the next month of work.
Run before committing to the sweep.


---

## 4e. Why cross-subject transfer fails — a four-experiment diagnosis

Six of 23 LOSO folds sit at chance. The obvious hypotheses were tested in
increasing cost order, and all but the last were eliminated.

### Experiment 1 — is it model capacity?

Random forest on the four worst folds, against logistic regression:

| fold | logreg | RF (100 trees) |
|---|---|---|
| chb06 | 2.8× | 1.7× |
| chb12 | 1.3× | 2.4× |
| chb13 | 1.1× | **0.8×** |
| chb15 | 1.7× | **0.9×** |

**No.** RF gave 156× on chb01 with the same features and training data — a model
with the capacity to reach 156× elsewhere is not failing here for lack of
expressiveness. chb13 and chb15 land *below* chance (lift < 1.0), meaning the
ranking is slightly anti-correlated with truth.

Note this also rules out decision-boundary geometry as the cause: RF (axis-aligned
partitions) and logistic regression (a global hyperplane) have opposite inductive
biases and fail identically. A third model class would not have been informative.

### Experiment 2 — do the features work at all?

Patient-specific RF on chb15, leave-one-record-out (whole records held out, so no
window overlap across the split):

| held-out record | lift |
|---|---|
| chb15_06 | 26.2× |
| chb15_10 | **107.3×** |
| chb15_15 | 22.0× |
| chb15_17 | 77.7× |
| chb15_20 | 56.2× |

**Yes — median ~56× against a cross-subject 0.9×.** This rules out three things at
once: the features capture chb15's seizures, the labels are correct (you cannot
get 107× against wrong labels), and it is not model capacity.

**This gives a ceiling.** Every generalization technique from here is measured as
*what fraction of the within-subject ceiling it recovers* — a better target than
raw AUPRC, and one that normalizes across subjects of differing difficulty.

### Experiment 3 — is it scale/offset mismatch?

Within-**record** z-scoring (median-centred, so ictal windows do not shift the
baseline), then re-run logistic regression:

| fold | unnormalized | normalized |
|---|---|---|
| chb06 | 2.8× | 1.3× |
| chb12 | 1.3× | 1.3× |
| chb13 | 1.1× | 1.5× |
| chb15 | 1.7× | 2.1× (sens 0.05 → 0.25) |

**No.** Against chb15's 56× ceiling, 2.1× recovers about **4% of the gap**.
Amplitudes being shifted or stretched relative to the population is not the
problem.

### Experiment 4 — is the discriminative direction itself patient-specific?

Compare the coefficient vectors of a chb15-only model and a population model
(both with the scaler inside the pipeline, so both operate on standardized
features):

```
coefficient correlation : r = 0.106
top-20 feature overlap  : 3 / 20
```

**Yes.** The two models weight essentially unrelated features. No transform of
feature *values* fixes this, because the issue is *which* features matter.

chb15's top features:

```
FP1-F3__dwt_cD1   FP1-F3__hjorth_activity
F3-C3__dwt_cD1    F3-C3__dwt_cD2
C3-P3__dwt_cD1    P3-O1__dwt_cD1
P3-O1__hjorth_complexity        F8-T8__hjorth_activity
```

**Five of eight are the complete left parasagittal chain** (FP1-F3, F3-C3, C3-P3,
P3-O1). A left-sided focus is a plausible, interpretable localization signal —
exactly what a patient-specific model can learn and a population model cannot.

### Resolved: `dwt_cD1` is not carrying the result (D42)

`dwt_cD1` covers **64–128 Hz — entirely inside the 0.5–40 Hz stopband.** Two
readings, and they have very different consequences:

- **Real high-frequency content.** The lowpass is 2nd-order Butterworth
  (~24 dB/decade), so gamma and ripple energy is attenuated, not eliminated.
  High-frequency oscillations are a documented seizure biomarker, and residual
  energy above 40 Hz could genuinely be the most discriminative signal available.
- **Filter-rolloff artifact.** If cD1 is mostly noise, the within-subject model may
  be fitting a recording-session signature rather than physiology — which would
  inflate the 56× ceiling.

**Test:** refit within-subject chb15 with all 36 `dwt_cD1` and `dwt_cD2` features
dropped (291 → 255).

| record | with cD1/cD2 | without | retained |
|---|---|---|---|
| chb15_06 | 26.2× | 23.5× | 90% |
| chb15_10 | 107.3× | 96.1× | 90% |
| chb15_15 | 22.0× | 21.8× | 99% |
| chb15_17 | 77.7× | 36.2× | **47%** |
| chb15_20 | 56.2× | 53.8× | 96% |

**The ceiling survives.** Four of five folds retain 90–99%; the median drops
56.2 → 36.2×. The within-subject result is physiological, not stopband artifact.

*Caveat:* this run used `logreg` while the original used `rf`, so part of the gap
is model class rather than the dropped features. With four folds above 90% the
conclusion is unaffected.

`chb15_17` is the outlier at 47% — that single record leaned hard on
high-frequency content. Either genuine HFO activity in that seizure or a
session-specific artifact. One record of five; noted, not chased.

**Consequences.** For D22, the case for 128 Hz decimation *strengthens*: cD1
contributes ~10% of within-subject performance on most records and costs 36
features plus compute. Fold it into the feature ablation rather than treating it
as a separate decision. Separately, **255 features retaining ~90% of performance
is the first ablation data point** for the feature-budget curve — directionally
consistent with the op-count analysis that flagged the DWT as the most expensive
family covering the least aligned bands.

### What this means

The finding *is* the result. Cross-subject seizure detection on CHB-MIT is
**bimodal**, and the failing subjects are fully recoverable from minutes of their
own data. That is the deployment story — and it is precisely why RNS-class
devices are tuned per patient rather than shipped with a population model.


---

## 4f. Patient calibration curve

**The core result.** How much of a patient's own data does the model need before
it works?

Design: population model trained on the other 22 subject groups (10% interictal
subsample, all positives). The patient's seizure records are split
chronologically — the last 40% are a **fixed test set**, never touched; the
earlier 60% form a pool. At each *n*, the first *n* pool records join training
(sample-weighted 5×), and the **remaining pool records calibrate the detection
threshold**. Test data never influences the threshold.

### Results (`target_fa = 2.0`, `max_duty = 0.01`, quantile thresholds)

| subject | n=0 | n=1 | n=2 | n=3 | n=5 | latency n=0 → best |
|---|---|---|---|---|---|---|
| **chb15** | 0.00 | 0.17 | 0.17 | 0.75 | **0.92** | — → 15 s |
| **chb12** | 0.00 | 0.16 | 0.21 | 0.47 | **0.68** | — → 18 s |
| **chb13** | 0.20 | 0.40 | 0.20 | **0.60** | — | 42 s → 22 s |
| **chb06** | 0.00 | 0.67 | 0.67 | 0.00 † | — | — → 4.5 s |
| chb09 *(control)* | **1.00** | — | — | — | — | 9 s |
| chb10 *(control)* | **1.00** | 1.00 | 1.00 | 1.00 | — | 14 → 13 s |

Event sensitivity. FA/h stayed in 0.0–3.8 across every row. AUPRC lift over the
same range: chb15 1.5 → 9.3×, chb12 1.3 → 7.9×, chb13 2.0 → 12.6×,
chb06 3.4 → 21.9×; controls 55–70× flat.

† chb06 n=3 is an artifact of calibration-data exhaustion — see limitations.

### What it shows

**Patient calibration is a targeted rescue, not a general improvement.** Every
hard subject starts at chance (1.3–3.4× lift, 0.00 sensitivity) and rises
monotonically. Both controls start at 55–59× with sensitivity 1.00 and stay
there. There is no overlap between the two groups at n=0 — the bimodality seen in
the LOSO folds is confirmed under a fixed test set.

The controls do gain in *ranking* (chb10: 54.9 → 69.7×) while sensitivity stays
pinned at 1.00. So patient data helps everyone; it only *matters* for the
subjects who need it.

**chb15 is the clearest case:** 0.00 → 0.92 event sensitivity at 3.02 FA/h, with
detection latency falling 83 s → 15 s, from five of the patient's own seizures.

---

## 4g. Two methodological findings

Both were discovered by the harness catching its own failures, and both are
non-obvious enough to be worth reporting.

### Absolute detection thresholds do not transfer across sessions

Selecting a threshold on a calibration record and applying that **numeric value**
to a test record fails badly. chb12 passed calibration at duty ≤ 0.02 and then
alarmed **25.9% of the time** on test — same model, same threshold, tenfold
different duty.

Cause: per-record score distributions shift with session amplitude and noise, so
an absolute threshold means something different on each recording, even within
one patient.

**Fix: transfer the quantile, not the threshold.** Select *q* on the calibration
records, then apply `np.quantile(test_scores, q)` at evaluation.

| | median test duty | **max test duty** |
|---|---|---|
| absolute threshold | 0.0096 | **0.0903** |
| **quantile transfer** | 0.0028 | **0.0071** |

FA/h across all runs went from a 0.0–22.9 range to 0.0–3.8.

This is also the *deployable* form: a device maintains a running score
distribution and alarms at a fixed percentile of its own recent history —
causal, and it adapts to session drift automatically.

### Duty-cycle monitoring is required, or the failure is invisible

chb12's pathological run scored **sensitivity 1.00** at 22.9 FA/h. Refractory
merging collapses continuous alarming into a small number of "events," so FA/h
alone made an always-on detector look merely noisy. `alarm_duty` = 0.259 exposed
it immediately. This is the practical vindication of D31.

### Selection-criterion ordering

FA/h estimated from one calibration record is computed from a handful of merged
events and is very noisy; duty is estimated from ~3,600 windows and is stable.
Constraining on duty first, FA/h second, is what removed the non-monotonic
sensitivity swings (chb15 previously ran 0.50 → 0.08 → 1.00 across consecutive
*n*).

In the final configuration `target_fa = 2.0` binds first, so `max_duty` values
above 0.01 are inert — the 0.02 and 0.05 sweeps returned byte-identical tables.
Duty is retained as a safety net.

---

## 4h. Limitations of the calibration experiment

**Calibration and training data compete for the same pool.** As *n* grows, fewer
records remain to calibrate the threshold. chb06 at n=3 calibrates on a single
record, selects q = 0.9944 against 0.9867/0.9918 at lower *n*, and drops to
sensitivity 0.00 despite the **highest** lift of its series (21.9×). The ranking
improved; the operating point did not transfer. A larger pool, or a dedicated
calibration hold-out, would remove this.

**Small test-event counts.** chb06 and chb10 test on 3 events, chb09 on 3,
chb13 on 5. Sensitivity 1.00 from 3 events is thin. chb12 (19 events) and chb15
(12) are the better-powered rows and should carry the argument.

**A shrinking test set inflates results ~3×.** An earlier design used
`test_recs = sz_records[n:]`, so the test set shrank as training data grew. Same
subject, same subsample, chb15 at n=1: **7.9× with a shrinking test set vs 2.6×
with a fixed one.** The consumed records were the chronologically earliest, and
the remainder differed systematically. This is a subtle enough trap to be worth
flagging on its own — it silently steepens any calibration curve built this way.

**Population subsample matters.** chb15 n=1 gave 7.9× at `subsample=0.1` and
4.4× at `0.03`. Population data contributes real signal, so the n=0 anchor is
sensitive to how much of it is used. All reported results use 0.1, consistent
with the LOSO baselines.


---

## 4i. 1D-TCN over feature sequences

Input is `(batch, 291 features, T timesteps)`, one timestep per 2 s window at 1 s
stride — **not raw EEG**. Windows are ~50× the storage of features and the
feature pipeline is already validated and frozen.

### Causality is verified, not assumed

Every convolution is left-padded by `(kernel-1) * dilation` with the right end
trimmed. PyTorch's `Conv1d` pads symmetrically, which leaks the future.

**A non-causal TCN trains beautifully and is undeployable, and nothing in a loss
curve reveals it.** Two tests, both returning `0.000e+00`:

| test | method | result |
|---|---|---|
| causality | perturb one timestep by +100, check no *earlier* output moves | **0.000e+00** |
| streaming equivalence | score a prefix, compare to the same slice of a full pass | **0.000e+00** |

The second is the sequence-level analogue of the filter equivalence test and
proves the model can run incrementally on a live stream.

### Compute budget — corrected

An initial estimate of ~93k MACs/timestep was **wrong**: a standard TCN residual
block has *two* convolutions plus a 1×1 downsample, not one. Measured:

| block | MACs/timestep |
|---|---|
| block 0 (291→64) | **86,784** |
| blocks 1–3 (64→64 each) | 24,576 |
| **total** | **160,576** |

That is **25% of the ~650k feature-extraction budget**, not 14%. Block 0 alone
is 54% of the model — the 291→64 projection plus its residual downsample is
74,496 of it.

| configuration | MACs/timestep | params | % of feature budget |
|---|---|---|---|
| 291 features, hidden=64 | 160,576 | 161,665 | 25% |
| 291 features, hidden=32 | 58,784 | 59,329 | 9% |
| **32 features (4 ch × 8), hidden=32** | **24,608** | **25,121** | **4%** |

A 6.5× reduction at an unchanged 61-timestep receptive field. **This is the
second, sharper motivation for the feature-budget curve: input dimension
dominates the model itself, not just extraction.**

Receptive field is 61 timesteps (61 s at 1 s stride) — again double the initial
estimate, for the same two-convs-per-block reason.

### Within-subject: 18.8×, and data-limited

chb15 trained on its own records, three configurations:

| T | stride | warmup | lift |
|---|---|---|---|
| 96 | 48 | none (bug) | 18.54× |
| 96 | 48 | 48 | 18.52× |
| 192 | 48 | 61 | 18.82× |

All three land at 18.5–18.8× with validation loss bottoming at **epoch 1–2**.
Neither sequence length nor warmup masking moved it. AUROC is 0.978–0.980, so
ranking is excellent and the loss is in precision near the decision boundary.

The binding constraint is **~17 seizure events** in chb15's training records. No
architectural change creates more. Against the classical within-subject ceiling
of 36–56×, the TCN reaches roughly half.

### Cross-subject: below chance

Population pretraining on 22 subject groups, chb15 held out entirely:
**342,519 windows, 178 events** — 10.5× chb15 alone. Training was healthy in a
way per-subject training never was: validation improved on 19 of 20 epochs in the
first run and converged over 60 total epochs to 0.0039, no overfitting.

Evaluated on chb15's held-out records:

```
AUROC 0.411   AUPRC lift 0.79×
```

**Below chance.** Not a failure to transfer — the model ranks ictal windows
*lower* than interictal ones for this subject. Pipeline correctness confirmed by
scoring chb10 (which *was* in training): AUROC 0.99996, lift 94.2× — memorization,
but it proves the scoring path and sign convention are right.

Negating the scores gives lift 4.35×, so the trunk learned real structure that is
**systematically inverted** for chb15. That is a more precise claim than "it
failed", and it matches the r = 0.106 feature-level finding at the representation
level.

### Fine-tuning: head-only fails, full fine-tuning partially recovers

| approach | trainable params | lift |
|---|---|---|
| population only | — | 0.79× |
| + head fine-tune (LR 1e-3) | 33 | 0.72×, *falling* to 0.67× |
| + head fine-tune (LR 1e-5) | 33 | 0.72×, flat |
| + **full fine-tune (LR 1e-4)** | 59,329 | **14.28×** and still rising |
| chb15-only from scratch | 59,329 | 18.8× |

**33 parameters cannot recombine an inverted representation into a correct one.**
The head is a 1×1 conv over 32 channels; it can rescale and mix, not repair.

A bias-collapse hypothesis was tested and rejected: AUROC is rank-based and
therefore *exactly* invariant to an additive shift. Verified — shifting every
logit by −100 left AUROC unchanged at 0.32014 to all printed digits. The damage
came from the 32 weights overfitting 408 ictal windows, not from the bias.

**Outstanding:** whether pretraining contributes anything at all, tested by
running full fine-tuning from a random initialization under an identical
protocol. If random init matches the pretrained one at plateau, the pretrained
weights are functioning as initialization only and the transfer claim is empty.


---

## 4j. TCN per-subject results — the deployable configuration

Population pretraining showed **negative transfer** (below), so every model here
is trained per subject from random initialization. Protocol matches the classical
calibration curve exactly: seizure-bearing records only, first 5 for training and
threshold calibration, last 3 held out. Thresholds by quantile transfer (D44),
selected on calibration records only.

Medians over three seeds:

| subject | AUPRC lift | sens | FA/h | latency | logreg sens @ FA/h |
|---|---|---|---|---|---|
| **chb06** | 538.6× | **1.00** | 1.60 | **5 s** | 0.67 @ 3.67 |
| chb09 *(ctrl)* | 114.5× | 1.00 | 0.54 | **5 s** | 1.00 @ 1.11 |
| chb10 *(ctrl)* | 92.3× | 1.00 | 1.04 | **6 s** | 1.00 @ 1.03 |
| **chb12** | 21.2× | **0.91** | 0.41 | 13.5 s | 0.68 @ 1.67 |
| chb13 | 17.7× | 0.60 | 1.49 | 18 s | 0.60 @ 0.73 |
| chb15 | 14.8× | 0.86 | 0.40 | 21 s | 0.92 @ 3.02 |

**The TCN wins on five of six subjects and on latency everywhere.** chb06 goes
from 0.67 sensitivity at 3.67 FA/h to 1.00 at 1.60, detecting in 5 s. chb12 goes
0.68 → 0.91 while cutting false alarms 4×. Four of six now detect within 6 s
against a 4 s architectural floor (window length + 3-of-5 persistence).

**Seed stability is excellent:** sensitivity has σ = 0.0 on every subject across
three seeds — identical detections every time. Lift σ < 0.5 on five of six. The
exception is chb06 (σ = 71.6 on a median of 538.6×), but that is volatility in a
ratio with a 0.121% denominator, not in the detector: its sensitivity, FA/h and
latency are all stable.

**Caveats.** chb06, chb09 and chb10 rest on 3 test events, so sensitivity 1.00
has a 95% CI of roughly [0.44, 1.00]. chb12 (11 events) and chb15 (7) carry the
argument. chb13 is the one subject that does not benefit — AUROC 0.75 against
everyone else's 0.94–0.999, and sensitivity 0.60 matching logreg exactly.

---

## 4k. Streaming fixed-point inference — validated

### Streaming, not recompute

Re-running a 192-step window for each new timestep does **192× the necessary
work**. A causal dilated conv at timestep *t* needs only `(k-1)·dilation` past
activations, so each layer keeps a ring buffer and computes one output column.

| | per-window MACs | activation cache |
|---|---|---|
| naive recompute | 11,286,528 | none |
| **streaming** | **58,784** | 2,953 values = **5.8 KB int16** |

Note the input buffer alone (291 features × 3 taps = 873 values) is **30% of the
cache** — a third argument for feature reduction, alongside parameters and MACs.

Correctness is the same contract the causal filter satisfies:

| test | result |
|---|---|
| streaming vs batch, synthetic | 2.13e-07 |
| streaming vs batch, trained model | 3.72e-07 |
| **streaming vs batch, real EEG features** | **1.53e-06** |
| reset isolation (two runs after reset) | **0.000e+00** |

Wall clock 0.175 ms p50 in interpreted NumPy float64. **The budget is 1000 ms,
not 15.6 ms** — an earlier framing conflated two rates. The TCN runs once per
*window* (1 s stride); only the filter runs per sample. Compute was never the
constraint.

### Q15 quantization

Weights int16 at **per-layer power-of-two scales**, so dequantization is a shift
rather than a multiply. Per-layer rather than global because `blocks.0.c1` peaks
at |w| = 0.0637 while later layers reach 0.13 — one global scale would waste a
bit of precision.

**The accumulator is the real constraint, not the stored values.** Precision-optimal
shifts (17–18) put the worst-case accumulator at 33–36 bits, overflowing int32 on
*every* layer. `blocks.0.c1` sums 873 products and is worst at 35.9 bits.

Fix: **reduce the weight shift to buy accumulator headroom.**

| | precision-optimal | safe | used | acc bits | rel err |
|---|---|---|---|---|---|
| `blocks.0.c1` | 18 | 12 | 12 | 29.9 | 1.92e-03 |
| `blocks.0.down` | 18 | 12 | 12 | 29.3 | 1.34e-03 |
| all others | 17 | 13 | 13 | ~29.5 | ~4.6e-04 |

Five bits of weight precision bought six bits of headroom, at a cost of ~1e-03
relative error — well under the ~1e-02 where AUPRC would move.

The worst case is deliberately pessimistic (every input at maximum, all signs
aligned with the weights), which never occurs. Designing to it is still correct:
**overflow is silent and produces plausible garbage.**

GELU is the only transcendental. A 512-point lookup table with linear
interpolation gives **9.77e-05** max error — negligible against the 1.9e-03
weight error, and far cheaper than an erf evaluation on a DSP without an FPU.

### Q15 costs zero detection performance

Scored the full held-out set (9,783 windows, 7 events) through both paths, using
the **same quantile** for each — the deployable threshold rule (D44):

| metric | float64 | Q15 | delta |
|---|---|---|---|
| AUPRC lift | 15.5997 | 15.6019 | +0.0022 |
| AUROC | 0.9902 | 0.9902 | 0.0000 |
| sensitivity @ q=0.95 | 0.8571 | 0.8571 | **0.0000** |
| FA/h @ q=0.95 | 2.3501 | 2.3501 | **0.0000** |
| alarm duty @ q=0.95 | 0.0015 | 0.0015 | **0.0000** |
| latency @ q=0.95 | 9.5 s | 9.5 s | 0.0 |
| sensitivity @ q=0.98 | 0.8571 | 0.8571 | **0.0000** |
| latency @ q=0.98 | 24.0 s | 22.5 s | **−1.5 s** |
| sensitivity @ q=0.99 | 0.7143 | 0.7143 | **0.0000** |

**Sensitivity, FA/h and alarm duty are identical at every operating point.** The
sole difference anywhere is 1.5 s of median latency at q=0.98, and it went
*down* — a single borderline window crossing threshold earlier under
quantization, not systematic degradation. Rank correlation between the two score
series is 0.999999.

Max absolute score error is 0.0158 against a range of 11.2 — **0.14%**. Because
the detector thresholds on a quantile, a uniform offset costs nothing; only
reordering would hurt, and there is essentially none.

### Deployment footprint

| component | size |
|---|---|
| TCN weights (int16) | 57.9 KB |
| activation cache (int16) | 5.8 KB |
| feature ring buffer (18 ch × 512, int16) | 18.4 KB |
| filter state | negligible |
| **total** | **~82 KB** |

Fits in flash on any Cortex-M4 and in SRAM on an M7. For comparison: random
forest at 100 trees is ~340 KB and does *not* fit; logistic regression is 1.2 KB.

Smaller configurations, same 61-timestep receptive field:

| config | params | int8 | MACs/step |
|---|---|---|---|
| 291 feat, h=32 *(deployed)* | 59,329 | 57.9 KB | 58,784 |
| 32 feat, h=32 | 25,121 | 24.5 KB | 24,608 |
| 32 feat, h=16 | 7,713 | 7.5 KB | 7,440 |


---

## 4l. Figures

Six figures, generated by `figures_bids.py` (publication styling + BIDS
derivative output) and `make_figures.py` (the earlier set). Rendered at 600 dpi
raster and vector PDF.

| # | figure | shows | status |
|---|---|---|---|
| 1 | calibration curve | sensitivity / latency / lift vs *n* patient seizures, hard subjects vs controls | code ready |
| 2 | LOSO bimodality | 23 folds sorted by lift with the chance line; the sensitivity–lift scatter showing no middle ground | code ready |
| 3 | threshold transfer | absolute vs quantile alarm duty (max 0.0903 → 0.0071) | code ready |
| 4 | corpus composition | events per subject, windows per event, class imbalance | code ready |
| 5 | **TCN vs logreg** | sensitivity, FA/h and latency across six subjects | **rendered** |
| 6 | **float vs Q15** | residual plot and identical operating points | **rendered** |

### Publication standards applied

- **Type 42 (TrueType) fonts embedded**, not Type 3 — many publishers reject
  Type 3 outright. Verify with `pdffonts`; it must report `CID TrueType`,
  `emb yes`.
- **Okabe–Ito colorblind-safe palette**, assigned semantically: hard subjects
  warm, controls cool. In figure 1 the separation between those groups *is* the
  finding, so it should not depend on parsing a legend.
- **Journal column widths** — 85 mm single, 180 mm double.
- **Vector PDF + 600 dpi PNG**, `svg.fonttype='none'` so SVG text stays
  editable rather than becoming paths.
- Bold A/B/C panel labels; minimum 6.5 pt type.

### Two honesty constraints in the figures themselves

**Sample size is annotated on the figure, not left to the caption.** Each legend
entry carries its test-event count — chb12 (11 ev) versus chb09 (3 ev) — so a
reader sees immediately which lines carry weight.

**Known artifacts are marked, not deleted.** chb06 at *n*=3 is circled and
labelled "calibration artifact" in figure 1. Unmarked it reads as a real
collapse, when in fact that point has the *highest* AUPRC lift of its series
(21.9×) and failed only because the threshold quantile was selected on a single
leftover calibration record.

A title correction was also required: an early draft of figure 5 read "TCN wins
on 5 of 6", which is false of the sensitivity panel — the TCN is higher on two,
tied on three, and *lower* on chb15 (0.86 vs 0.92). The win is the combination,
and the panel titles now state exactly what each panel shows.

### BIDS derivative organization

Figures are written under a BIDS-compliant derivatives root with
`dataset_description.json` (`DatasetType: derivative`, a `GeneratedBy` block,
and `SourceDatasets` pointing to the CHB-MIT DOI), entity-based filenames
(`desc-<label>_figure.pdf`), and a JSON sidecar per figure carrying its caption
and creation date.

```
derivatives/chbmit-lowlatency/
  dataset_description.json
  figures/
    desc-tcnVsLogreg_figure.pdf
    desc-tcnVsLogreg_figure.png
    desc-tcnVsLogreg_figure.json     <- caption + provenance
    desc-floatVsQ15_figure.{pdf,png,json}
```

**BIDS is a data organization standard, not a figure standard** — it specifies
naming, layout and metadata, and says nothing about how a plot looks. The two
requirements are independent and both are applied here. What BIDS buys is that
the output is a *reusable derivative dataset* rather than a folder of PNGs,
which is the concrete first step toward the BIDS-EEG derivative of the
harmonized corpus.


---

## 5. Decision log

Reasoning is recorded here because it is the part that becomes hard to
reconstruct later.

| # | Decision | Reasoning |
|---|---|---|
| D1 | S3 mirror over HTTP recursive fetch | ~161 KB/s vs. minutes; resumable |
| D2 | Manifest byte-diff as the integrity check | Size heuristics and header recomputation both produced false positives on this corpus |
| D3 | 18-channel canonical montage | Largest set consistently present; avoids subject-dependent feature shape |
| D4 | Select channels by name, not drop by name | 12 montages collapse to one under name selection; extras become irrelevant |
| D5 | Drop `P7-T7` | Polarity inverse of `T7-P7`; collinear |
| D6 | Regex-anchored parser + count assertion | Delimiter split silently dropped 12 seizures; warnings scroll away, assertions raise |
| D7 | Exclude 10 unannotated `chb24` records | Unannotated ≠ seizure-free; protects both label integrity and the FA/h denominator |
| D8 | Re-derive rather than drop 3 `chb12` files | Recovers 13 seizures (~⅓ of the patient); derivation is exact for `_27` |
| D9 | CAR / Laplacian **off** by default | Bipolar chain montage is already a spatial derivative; CAR assumes referential recording. Retained as an ablation flag |
| D10 | Aperiodic slope via OLS log-log, not FOOOF | Closed-form; pseudo-inverse is constant. 0.278 ms p50 — real-time viable. Labelled a surrogate |
| D11 | 0.5 Hz high-pass corner despite 35 ms group delay | Raising the corner *increases* delta-band delay (73 ms @ 1.0 Hz, 123 ms @ 2.0 Hz). Group delay ≠ compute latency |
| D12 | Report AUPRC / sensitivity@FA-per-hour / detection latency; **never accuracy** | ~0.3% positive rate makes accuracy ~99.7% for a null model |
| D13 | Two-stage build: cache features to Parquet | Extraction is the expensive step; model fitting is not. Never re-extract to try a classifier |
| D14 | Retain superseded checks in the notebook, labelled | Which checks were wrong and why is more informative than only showing the ones that worked |
| D15 | Accept the chb12 re-derivation as validated | Frequency-dependent (not flat) gap rules out scaling error; harmonic peak structure identifies an instrumental source; per-channel spread shows no bad electrode |
| D16 | No global notch filter for the chb12 interference | Three files out of 686; a global filter makes every file pay compute and group delay. Ablation on the subset instead |
| D17 | Carry `derivation` as a covariate, not just a column | Interference peaks at 16.25/27.50/32.38/36.00 Hz land in beta and low gamma and will bias `relpow_beta` and `aperiodic_slope` on those files |
| D18 | Keep the 0.5–40 Hz passband unchanged | Excluding the in-band peaks would require cutting to ~15 Hz, discarding beta entirely — a bad trade, since seizure discharges have real beta content |
| D19 | Flag `chb12_27` by record, not all three by `derivation` | `_28`/`_29` fall inside the native power range; the artifact is session-specific, so a `derivation`-wide flag would be too coarse |
| D20 | No CAR or Laplacian matrix in the default path — settled | The bipolar montage *is* the static spatial transform. Bipolar chains telescope (`(FP1−F7)+(F7−T7)+(T7−P7)+(P7−O1) = FP1−O1`), so the cross-channel mean is a mix of chain endpoints with no anatomical meaning. Also costs 165,888 MACs/window (~25% of the feature budget) |
| D21 | Parameterize windows by **duration**, not sample count | Frequency resolution depends on window duration, not sampling rate — a 2 s window gives 0.5 Hz bins at both 256 and 128 Hz. Hardcoding `512` is what makes the rate decision expensive later |
| D22 | Defer the 256 vs 128 Hz decision to an ablation | The 0.5–40 Hz bandpass already serves as the anti-aliasing filter for a 2× decimation (40 < 64 Hz new Nyquist), so decimation is free *after* filtering. Whether anything above 40 Hz carries signal is empirical, not architectural |
| D23 | Work in microvolts, not volts | A fixed `_EPS = 1e-12` is a ~1% distortion on Hjorth mobility at EEG-in-volts scale. Epsilon guards must be scaled to the data |
| D24 | Per-subject Parquet shards, not one table | A disconnect costs one subject rather than an hour; LOSO folds load without materializing 4 GB; Drive sync is incremental |
| D25 | Assert rather than return `None` on missing metadata | An empty `summarized` dict silently produced 3 empty shards. Lookup misses that return a plausible "no data" are the failure mode that costs the most |
| D26 | **Report per-fold results, never the mean alone** | 44% of positives come from four subjects; chb02/07/11/19/22 have 3 events each. Fold variance is large and a mean hides it. Always print the per-fold table, the spread, and the worst fold |
| D27 | Glob by record prefix, not directory name | chb17's files live in `chb17/` but carry `chb17a_`/`b_`/`c_` prefixes. A directory glob silently produced 3 empty shards |
| D28 | Read Parquet with `columns=[...]` for inspection | `pd.read_parquet` on the features directory loads ~4 GB before pandas overhead and crashed the runtime |
| D29 | Subsample training folds only, never the test fold | k-of-n persistence and detection latency both require contiguous windows. A subsampled test fold makes both meaningless |
| D30 | Thread `subsample_rate` through the FA/h denominator | At keep-rate *r* each surviving negative represents 1/*r* seconds. Verified: r=0.1 moves interictal hours 0.99 → 9.89 and FA/h 1.01 → 0.10 |
| D31 | Report `alarm_duty` alongside FA/h | Refractory merging collapses continuous alarming into one event, so an always-on model scores a deceptively low FA/h. Verified: a random model posted sens 1.00 at 2.02 FA/h with duty 0.50 |
| D32 | Report the **median** lift, not the mean | Mean 52.6× vs median 28.6× — the distribution is skewed by four high-performing folds |
| D33 | Save per-fold results incrementally inside the CV loop | Hours of compute otherwise depend on one variable assignment surviving a disconnect |
| D34 | Random forest as the sole baseline; drop linear SVM | LogReg and LinearSVC agree to within 4% (35.7× vs 34.2×) — redundant. RF is 4.4× better AUPRC with lower FA/h and half the latency |
| D35 | Run RF on the four chance-level folds *before* the full sweep | Whether RF rescues chb06/12/13/15 determines whether the next step is a TCN (capacity problem) or per-subject normalization (feature/calibration problem). 2 h to answer |
| D36 | 1D-TCN consumes **feature sequences**, not raw EEG | Windows are ~50× the storage of features; re-extracting raw from 45 GB is not feasible on Colab. Feature sequences also isolate one variable: hand-designed features + learned temporal integration vs. RF's per-window independence |
| D37 | Build the fixed-point edge path in parallel with model selection | The expensive on-device work is feature extraction (~650k weighted ops), which is already frozen. The model is a dot product or tree traversal by comparison |
| D38 | Stop optimizing pure cross-subject transfer | Four experiments eliminated capacity, decision-boundary geometry, and scale mismatch. The discriminative direction is patient-specific (r = 0.106, 3/20 overlap) — no feature transform fixes that |
| D39 | Target architecture is population pretraining + patient-specific calibration | Matches the physiology, matches how shipped devices work, and the within-subject ceiling (22–107× on chb15) shows the headroom is real |
| D40 | Measure generalization as **fraction of the within-subject ceiling recovered** | Raw AUPRC is not comparable across subjects of differing difficulty. The ceiling normalizes it |
| D41 | Report the calibration curve — performance vs. minutes of patient data | This is a device specification, not a leaderboard number: it tells a clinician how long before the device is useful on a new patient |
| D42 | `dwt_cD1`/`cD2` verified as non-load-bearing — ceiling is physiological | Dropping all 36 features (291→255) retains 90–99% of within-subject lift on 4 of 5 records. Strengthens the case for 128 Hz decimation (D22) and gives the first feature-ablation data point |
| D43 | Fixed test set: hold out the last 40% of seizure records permanently | A shrinking test set inflated results ~3× (chb15 n=1: 7.9× → 2.6×). Every *n* must be scored on the same events or the curve is not a curve |
| D44 | **Transfer the quantile, not the threshold** | Absolute thresholds do not survive session change: chb12 calibrated at duty ≤ 0.02 and hit 0.259 on test. Quantile transfer cut max duty from 0.0903 to 0.0071. Also the deployable form — a device tracks a running percentile of its own history |
| D45 | Constrain threshold selection on duty first, FA/h second | FA/h from one calibration record comes from a handful of merged events and is very noisy; duty comes from ~3,600 windows. Duty-first removed the non-monotonic sensitivity swings |
| D46 | Cache model scores; tune thresholds offline | Fitting is the expensive deterministic step, threshold selection is cheap and tunable. Caching turned each 40-minute parameter sweep into seconds |
| D47 | Calibrate on **all** unused pool records, not just the next one | One record gave a high-variance threshold estimate — chb15 swung 0.50 → 0.08 → 1.00 across consecutive *n* purely from calibration noise |
| D48 | TCN consumes feature sequences, not raw EEG | Windows are ~50× the storage of features; re-extraction from 45 GB is infeasible here. Feature sequences also isolate one variable: hand-designed features + learned temporal integration vs. per-window independence |
| D49 | Verify causality by perturbation, every time the architecture changes | A non-causal TCN trains well and is undeployable, and no loss curve reveals it. Both tests return 0.000e+00 |
| D50 | Stream sequences from a flat matrix, do not materialize them | Materializing costs T/stride copies of the feature table — chb09 alone produced 4.5 GB and crashed the runtime. Indexing made population-scale training possible at 0.40 GB |
| D51 | Cap `pos_weight` at ~50, do not use the theoretical value | At chb09's 0.098% imbalance the computed weight was 1024.5, which produced catastrophic memorization (train 0.0135, val 35.5). The loss surface becomes dominated by a handful of samples |
| D52 | Select training subjects by **seizure-bearing records**, not all records | `recs[:5]` picked five seizure-free records of chb15's 40, so a fine-tuning run trained on 0.000% ictal and the loss decreased smoothly while learning nothing |
| D53 | Evaluate per epoch with `window_metrics`, not loss alone | Loss is not comparable across `pos_weight` settings and is not the objective. Head fine-tuning showed loss falling 53.7 → 18.3 while AUROC fell 0.344 → 0.320 |
| D54 | Train the TCN **per subject from random init** — no population pretraining | Pretraining showed *negative* transfer: random init scored 5.98× at epoch 0, beating the pretrained model's entire population training (0.77×), and plateaued higher (15.6× vs 14.8×, AUROC 0.992 vs 0.939). The model must first unlearn an inverted representation |
| D55 | Stream with cached dilated activations, never recompute the window | 58,784 MACs vs 11,286,528 — a 192× reduction for a 5.8 KB cache. On an implantable this is the difference between viable and not: energy per decision × duty cycle = battery life |
| D56 | The TCN budget is **1000 ms**, not 15.6 ms | An earlier framing conflated two rates. The TCN runs once per *window* (1 s stride); only the filter runs per sample. Measured 0.175 ms p50 — 0.02% of budget |
| D57 | Choose Q15 weight shifts by **accumulator headroom**, not precision | Precision-optimal shifts (17–18) overflow int32 on every layer (33–36 bits). Dropping to 12–13 costs ~1e-03 relative error and fits in 29.5 bits. Overflow is silent and produces plausible garbage |
| D58 | GELU by 512-point lookup with linear interpolation | 9.77e-05 max error — negligible against the 1.9e-03 weight error, and far cheaper than erf on a DSP with no FPU. It is the only transcendental in the network |
| D59 | Drop weight-norm from the deployment architecture | It reparameterizes the optimization and changes nothing about the trained function, but its parametrization machinery broke three separate times and does not survive export. `PlainSeizureTCN` is the deployment form |
| D60 | Validate quantization on **detection metrics**, not score error | 0.14% score error and 0.999999 correlation predict no change; measuring confirmed sensitivity, FA/h and duty are identical at every operating point. Predicting is not measuring |
| D61 | Embed fonts as Type 42, never Type 3 | Many publishers reject Type 3 outright. `pdf.fonttype=42`, `ps.fonttype=42`, `svg.fonttype='none'`; verify with `pdffonts` |
| D62 | Annotate sample size **on the figure**, not in the caption | chb09 rests on 3 test events and chb12 on 19. A reader should see which lines carry weight without hunting for it |
| D63 | Mark known artifacts rather than deleting them | chb06 *n*=3 is circled as a calibration artifact. Unmarked it reads as a real collapse, when that point has the highest lift of its series |
| D64 | Panel titles state what the panel shows, not the headline claim | "TCN wins on 5 of 6" is false of the sensitivity panel alone (higher on 2, tied on 3, lower on 1). The win is the combination |
| D65 | Emit figures as BIDS derivatives with JSON sidecars | Makes the output a reusable derivative dataset rather than loose PNGs, and is the first concrete step toward a BIDS-EEG derivative of the harmonized corpus |

---

## 6. Leakage traps — identified, not yet implemented

To be enforced in the windowing/CV layer:

1. **`chb21` is `chb01`.** PhysioNet states case chb21 was recorded from the
   same subject as chb01, 1.5 years later. Grouping CV by directory name puts
   the same person on both sides of a split.
2. **`chb17a` / `chb17b` / `chb17c` are one patient.** A single
   `chb17-summary.txt` spans three directory prefixes. Any code inferring
   subject from the *summary filename* rather than the *record name* gets this
   wrong.
3. **Overlapping windows are near-duplicates.** A 2 s window at 1 s stride
   shares half its samples with its neighbour. Cross-subject splits handle this;
   within-subject protocols must hold out whole *records*, not individual
   seizures.
4. **Scalers must be fit inside the CV pipeline**, on training folds only.
   Fitting on the full table before splitting is the most common silent break in
   this kind of pipeline.

---

## 7. Compute budget

Measured on a reference implementation, **not yet reproduced in this
environment** — reproduce before citing.

Budget: one 4-sample packet at 256 Hz = **15.625 ms**.

| stage | p50 | p99 |
|---|---|---|
| causal bandpass, one packet | 0.042 ms | 0.101 ms |
| ring buffer shift | 0.005 ms | 0.011 ms |
| all 291 features | 1.084 ms | 1.492 ms |
| logistic regression, raw numpy dot | 0.005 ms | 0.016 ms |
| **total loop** | | **1.62 ms — 10% of budget** |

**Random forest does not fit, and the reason is instructive:**

| config | total nodes | p50 |
|---|---|---|
| 300 trees, unlimited depth | 84,296 | 21.7 ms |
| 300 trees, depth 8 | 17,096 | 20.4 ms |
| 50 trees, unlimited | 14,144 | 3.6 ms |
| 25 trees, depth 8 | 1,357 | 1.9 ms |

Constraining depth 5× barely moves the clock while tree count moves it linearly:
the cost is ~70 µs of per-tree dispatch overhead, not node traversal. Batching
256 windows drops it to 161 µs/window, confirming the same from the other
direction. → Report a **latency/accuracy trade curve**, not one number.

### The device gap

Shipping closed-loop devices are far more parsimonious than a research pipeline.
NeuroPace RNS thresholds a single preselected feature (signal intensity, line
length, or half-wave) on 4 channels. Medtronic sensing platforms run spectral
analysis plus a linear classifier on 4 channels with 2–8 features total.

| configuration | dim | weighted ops/window | buffer | ratio |
|---|---|---|---|---|
| research default, 18 ch × 16 | 288 | 649,602 | 18.4 KB | 1× |
| Medtronic-like, 4 ch, spectral + linear | 40 | 71,476 | 4.1 KB | 9× cheaper |
| RNS-like, 4 ch, Hjorth + line length, no FFT | 16 | 14,656 | 4.1 KB | 44× cheaper |

(Weighted proxy charges a transcendental at ~20 MACs — roughly a table-lookup
log on a fixed-point DSP with no FPU.)

**Two findings that invert the usual guidance:**

- **DWT is the most expensive family, not the cheapest.** A 4-level db4 Mallat
  tree costs 15,872 MACs/channel against 5,377 for one 512-point rFFT (~2.9×) —
  and the rFFT is *shared* across band power, spectral entropy, and the
  aperiodic slope, while the DWT serves only itself.
- **Transcendentals are the hidden line item.** Spectral entropy needs a log per
  frequency bin; Hjorth and line length need almost none. That is not a
  coincidence — it is why RNS-class detectors use line length.

### Two distinct latencies

- **Compute latency** — wall-clock per packet. 15.625 ms budget.
- **Detection latency** — seconds from annotated onset to alarm. Floored by
  window length plus any k-of-n persistence rule. No SIMD work reduces it.

Conflating these in a write-up is the kind of thing a reviewer catches.

---

## 8. Immediate next steps

Ordered by what unblocks the most, not by what is most interesting.

### 1. Email Roy — today

Attach the manuscript draft, not a summary. Three specific asks:

- **Is this preprint-worthy, and where?** He will know whether it fits
  *J. Neural Engineering*, *IEEE TBME*, or a workshop. A five-minute answer that
  saves months of guessing.
- **Can you or Aazhang sponsor CRC access?** Frame it with the number: the
  23-subject extension is ~17 GPU-hours across three seeds, which is infeasible
  on Colab.
- **Does the r = 0.106 result connect to SP-BAND?** Whether patient-specificity
  also appears in aperiodic parameters is a question he would have an opinion on,
  and it links independent work to the lab's.

This is twenty minutes and it potentially unblocks both the compute and the
faculty acknowledgment — the two things that most change how the project
reads.

### 2. Render the four remaining figures

Code and results exist for all of them. One hour. This is the gap between a
project that is impressive to *read* and one that is impressive to *skim*, and
skimming is what actually happens.

### 3. Preregister the 23-subject extension — before running it

Everything so far is retrospective and the manuscript says so. Specify the
protocol, the primary metric, the operating point and the exclusion rules.
Expect the act of writing it to surface at least two choices you had not
realised were choices; that is the value.

### 4. The 23-subject run

One seed for the 17 untested subjects, keeping three seeds on the six already
done — ~6 h, and defensible given that event sensitivity had zero variance
across seeds. If CRC access comes through, run all three seeds.

**Checkpoint every model.** The six-subject sweep was not saved and had to be
partly regenerated.

### 5. Study §4 of the guide while things run

Ben-David et al. (2010) is one paper and it is the largest gap between what has
been measured and what can be claimed. The bound decomposes target error into
source error, distributional divergence, and λ — the error of the best
joint hypothesis. Alignment methods reduce the divergence term; nothing reduces
λ. The r = 0.106 result is evidence that λ is large, which is exactly the
case the theory says cannot be fixed by reweighting or feature alignment.

### Explicitly deferred

KiCad, Riemannian recentering, the C port, and the feature-budget curve. All
worthwhile, none blocking, and all of them compete with finishing what is
started.


---

## 8b. Path to edge deployment

**The edge port does not wait on model selection.** The expensive on-device work
is feature extraction — ~650k weighted ops per window — and that pipeline is
frozen. The classifier is a dot product or a tree traversal by comparison. The
Q15 fixed-point path can be built in parallel with the RF sweep.

Estimated effort for a **simulated** target (Q15 in NumPy/C on desktop):

| task | hours |
|---|---|
| Q15 feature path — ring buffer, fixed-point filter, no dynamic allocation | 15–25 |
| Differential validation against the float path, per feature family | 8–12 |
| Model export — tree traversal in C, or a coefficient dot product | 5–10 |
| Latency and op-count harness on the fixed-point path | 5 |
| **total** | **35–50** |

Actual silicon (Cortex-M4/M7, CMSIS-DSP, toolchain, flashing) adds 40–80 hours
and requires hardware. The simulated version is the better artifact per hour: it
demonstrates the same engineering judgment without a supply chain.

The validation pattern is already established — the differential test that caught
the microvolt epsilon bug is exactly what a Q15-vs-float64 comparison needs, per
feature family, with a documented error bound.

**Expected findings worth planning for:** derivative-based features (line length,
both Hjorth parameters) will be the most sensitive to fixed-point precision loss,
since differencing amplifies quantization error. The aperiodic slope needs a
fixed-point log, which is where a table-lookup implementation earns its ~20-MAC
cost estimate.


---

## 9. Open questions

- **Detection or prediction?** Everything here is ictal-vs-interictal
  *detection*. Adding a preictal class is a small change to the labelling and a
  large change to the claim, with a different literature and credibility
  profile. Decide before the feature build, not after.
- **Does `derivation` matter?** Answered: no. Re-derivation introduces no
  detectable power shift; `chb12_27` alone carries session-specific
  interference, flagged by record (D19).
- **256 or 128 Hz for deployment?** Deferred to an ablation (D22). Extract at
  both, compare AUPRC against the halved op count.
- **Aperiodic surrogate justification.** Correlate the OLS log-log slope against
  offline FOOOF on a held-out window subset. Without that, the fast slope is an
  assertion. It is also *not* SP-BAND's knee-mode broadband fit — do not present
  them as interchangeable.
- **Preregistration.** OSF before running outcome analyses.

---

## Repository layout

```
CHB_MIT.ipynb                  main notebook, 16 sections
RESTORE_CELL.py                paste-at-top cell; rebuilds all state in ~30 s
figures_bids.py                publication styling + BIDS derivative output
make_figures.py                calibration curve, LOSO, threshold, corpus figures
manuscript_draft.{md,docx}     preprint draft
derivatives/chbmit-lowlatency/ BIDS derivative root (figures + sidecars)
chbmit_pipeline.py             standalone runnable pipeline (all stages)
data/record_manifest.csv       per-record inclusion decisions
data/features/chbNN.parquet    26 per-subject feature shards, 6.2 GB
README.md                      this file
```

### Running headless

```bash
python chbmit_pipeline.py --stage selftest    # no data needed, ~5 s
python chbmit_pipeline.py --stage manifest
python chbmit_pipeline.py --stage extract --drive /content/drive/MyDrive/chbmit/features
python chbmit_pipeline.py --stage verify      # asserts 198 events / 23 groups
```

`--stage extract` is idempotent — cached subjects are skipped, so a crash costs
one subject. `--stage selftest` validates the normalizer, subject grouping,
window labelling, stream equivalence, the extractor, the Hjorth differential
test, and `window_metrics`, with no data access.

Notebook sections: **0** setup and definitions (run first after any reconnect) ·
**1** acquisition and integrity · **2** montage survey · **3** annotation layer ·
**4** chb12 re-derivation · **5** canonical loader · **6** next steps.

Everything below Section 0 depends on Section 0 and nothing else.

---

## References

Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (v1.0.0). PhysioNet.
RRID:SCR_007345.

Goldberger, A., Amaral, L., Glass, L., et al. (2000). PhysioBank, PhysioToolkit,
and PhysioNet. *Circulation*, 101(23), e215–e220.

Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment.* PhD thesis, MIT.
