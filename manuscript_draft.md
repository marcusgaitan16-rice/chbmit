# Per-subject heterogeneity in cross-subject scalp EEG seizure detection, and a validated fixed-point streaming decoder

**[Author]**¹, **[Advisor]**²
¹ Department of Neuroscience, Rice University
² [Affiliation]

*Draft — not for circulation. Numbers current as of [date].*

---

## Abstract

Automated seizure detection on scalp EEG is usually evaluated either within
subjects, where performance is high, or across subjects, where it is
substantially lower. Reported cross-subject results vary widely, and pooled
summary statistics obscure how unevenly that deficit is distributed. In this
leave-one-subject-out protocol and feature space, cross-subject performance on
CHB-MIT showed an **apparent two-group distribution**: 6 of 23 subjects near
chance (AUPRC lift < 5×) and 17 reaching a median lift of 28.6×, with no
observed intermediate cases. Cross-subject detection does not fail universally
here — it fails for an identifiable minority.

Four experiments localize the cause. Model capacity is excluded — a random
forest fails identically to logistic regression on the same subjects despite
reaching 156× elsewhere, and the two have opposite inductive biases. Feature
adequacy is excluded — the same features reach 22–107× lift when trained
within a failing subject. Amplitude mismatch is excluded — within-record
z-scoring recovers only ~4% of the gap. What remains is examined directly:
the coefficient vectors of a patient-specific and a population model correlate
at *r* = 0.106 with 3 of 20 top features in common. Coefficient mismatch
together with the failed scaling ablations is **consistent with patient-specific
discriminative structure**; this mechanism is supported but not uniquely
identified, and representation-alignment methods remain untested.

We then quantify the remedy. A patient calibration curve — performance against
the number of the patient's own seizures used for fine-tuning — shows hard
subjects rising monotonically from chance (1.3–3.4× lift, 0.00 event
sensitivity) to 7.9–21.9× and 0.37–0.92 sensitivity with five records, while
control subjects remain flat at 1.00 sensitivity throughout. Patient
calibration is therefore a **targeted rescue**, not a general improvement.

Finally we deliver a detector validated for deployment. A patient-specific
causal temporal convolutional network reaches **event sensitivity 0.60–1.00 at
0.4–1.6 false alarms per hour with 5–21 s detection latency** across six
subjects — competitive with, but not exceeding, published patient-specific
CHB-MIT results, which commonly report 95–100% sensitivity at 0.08–0.66 FA/h
with 2–8 s delay under varying protocols and subject subsets. Our contribution
is not detection performance but **implementation validation**: the detector
runs as streaming fixed-point inference at 58,784 MACs per timestep in ~82 KB,
with verified causal-state equivalence, and Q15 quantization changes
sensitivity, false-alarm rate and alarm duty by exactly zero at every tested
operating point. Most CHB-MIT reports provide none of an operation budget, a
stateful stream-equivalence test, quantized-arithmetic validation, or a memory
footprint.

Two methodological findings emerge that generalize beyond this corpus:
**absolute detection thresholds do not transfer across recording sessions
within a patient**, while quantile-based thresholds do (maximum alarm duty
0.0903 → 0.0071); and **duty-cycle monitoring is required** to detect the
always-on failure mode, which false-alarm rate alone conceals.

---

## 1. Introduction

Responsive neurostimulation devices detect seizures on-device and deliver
therapy within seconds. The NeuroPace RNS thresholds a single preselected
feature — signal intensity, line length, or half-wave — on four channels;
Medtronic's sensing platforms run spectral analysis plus a linear classifier on
four channels with two to eight features in total. Both are **programmed per
patient** by a clinician using that patient's recorded events.

Most published seizure detection work optimizes a different objective: offline
classification accuracy on a benchmark, often within-subject, without a latency
or compute budget. This produces results that are difficult to interpret as
device performance.

We take the deployment constraint as the specification. Three commitments
follow:

1. **Every filter is causal.** No `filtfilt`, no future samples, including in
   offline dataset construction where non-causal filtering is easy to introduce
   accidentally and nearly impossible to detect downstream.
2. **Compute cost is a first-class metric**, budgeted in operations and
   transcendental calls rather than wall-clock, because wall-clock on a
   workstation does not transfer to a DSP.
3. **Evaluation is event-level.** Ictal windows are 0.339% of this corpus, so a
   null model scores 99.66% accuracy. Accuracy appears nowhere in this work.

Under these constraints we ask a question the deployment framing makes
unavoidable: *how much of a new patient's own data does a detector need before
it works on them?*

---

## 2. Methods

### 2.1 Data and inclusion

The CHB-MIT Scalp EEG Database (Guttag, 2010) comprises 686 EDF records from 23
pediatric subjects. All files were retrieved from the PhysioNet S3 mirror and
verified byte-exact against the remote manifest (686/686, zero mismatches).

Annotations were parsed from the per-subject summary files with a
regex-anchored parser that validates the number of parsed seizures against each
record's declared count and raises on mismatch. This validation was
load-bearing: the summary format mixes one- and two-space separators *within a
single record*, and a delimiter-based parser silently discarded 12 seizures
before the assertion was added.

**Corpus totals as parsed: 676 annotated records, 198 seizures, 3.23 h of ictal
time.**

Ten `chb24` records appear on disk with no summary entry and were **excluded**.
Unannotated is not seizure-free; including them as all-interictal would inject
unknown false negatives and inflate the interictal-hours denominator, depressing
the reported false-alarm rate.

### 2.2 Montage standardization

Twelve distinct channel montages appear across the corpus (22–38 channels). We
enforce a fixed 18-channel bipolar set: four chains of four (left and right
temporal, left and right parasagittal) plus `FZ-CZ` and `CZ-PZ`. `P7-T7` is
excluded as the polarity inverse of `T7-P7`; the `T7-FT9 / FT9-FT10 / FT10-T8`
chain is excluded because FT9/FT10 are not standard 10–20 positions and the
chain is inconsistently present, which would make the feature-vector shape
subject-dependent.

Under name-based selection, **683 of 686 files (99.6%) expose all 18 canonical
channels**. The 12 montages differ almost entirely in dummy (`--N`) and
non-canonical channels.

Three `chb12` records are not bipolar at all: one is referenced to CS2, two
carry bare electrode labels. These were re-derived rather than dropped,
recovering **13 seizures — approximately one third of that patient's events**.
Re-derivation is exact where the reference is shared (`FP1-F7 = (FP1-CS2) −
(F7-CS2)`); for the bare-label records a common reference is assumed and
recorded as an assumption. One header contains `01` (zero-one) as a typo for
`O1`, handled by an explicit alias.

Signal-quality validation of the re-derived records found their in-band power
inside the native range for two of three files, with the third (`chb12_27`)
carrying session-specific narrowband interference at 16.25, 27.50, 32.38 and
43.88 Hz. The 16.25/32.38 Hz pair is a fundamental and second harmonic,
indicating a periodic instrumental source rather than physiology.

### 2.3 Causal preprocessing

Second-order Butterworth bandpass, 0.5–40 Hz, applied forward-only via
`sosfilt` with filter state carried across blocks and reset per record.

**Stream equivalence is bit-identical.** Filtering a record in one pass versus
streaming it in 4-sample packets with state carried gives a maximum absolute
difference of **0.000e+00**. Two negative controls confirm the test is
meaningful: omitting state carry gives 2.091e-04 and `filtfilt` gives 4.445e-04,
both the same order as the signal itself (2.6e-04) and both silent.

Filter state is primed to the first sample of each record, reducing the
startup transient on a DC-offset signal by 27× (5.07e-05 → 1.88e-06 over the
first 64 samples, against a 1.49e-06 steady state).

Group delay is frequency-dependent and cannot be reduced below 15 ms in the
delta band: 35.1 ms at a 0.5 Hz high-pass corner, rising to 73.1 ms at 1.0 Hz
and 123.4 ms at 2.0 Hz. Raising the corner makes it worse. We therefore retain
0.5 Hz and note that group delay and compute latency are distinct quantities
that should not be conflated.

Spatial filtering (CAR, surface Laplacian) is **not** applied. The bipolar chain
montage is already the spatial transform: bipolar chains telescope
(`(FP1−F7)+(F7−T7)+(T7−P7)+(P7−O1) = FP1−O1`), so a cross-channel mean is a
mixture of chain endpoints with no anatomical interpretation. A dense 18×18
matrix operation would additionally cost 165,888 MACs per window, ~25% of the
feature budget.

### 2.4 Windowing and labelling

Two-second windows at one-second stride, parameterized by *duration* rather than
sample count so that frequency resolution (0.5 Hz) is invariant to sampling
rate. Windows overlapping any annotated seizure are labelled ictal.

**Windows within 60 s of a seizure boundary are dropped, not labelled
interictal.** A window ending three seconds before a marked onset is not a clean
negative, and CHB-MIT onset marks are reviewer estimates.

Three leakage controls are enforced. `chb21` is folded into `chb01` (PhysioNet
documents them as the same subject 1.5 years apart); `chb17a/b/c` are folded
into `chb17` (one summary spans three filename prefixes); and within-subject
protocols hold out whole *records*, since at 50% overlap consecutive windows
share half their samples. Feature scalers are fit inside cross-validation
pipelines on training folds only.

**Final table: 3,480,880 windows, 11,809 ictal (0.339%), 23 subject groups,
198/198 seizures preserved.**

### 2.5 Features

291 features per window: 18 channels × 16 plus three artifact guards. Per
channel: three Hjorth parameters, line length, five DWT log energies (db4,
4 levels), five relative band powers, normalized spectral entropy, and the
aperiodic spectral slope.

All constants are precomputed at construction — taper, frequency masks, DWT
filter bank, and the log-log regression pseudo-inverse.

The **aperiodic exponent** is computed as a closed-form ordinary least squares
fit of log₁₀(PSD) on log₁₀(*f*). Because the design matrix depends only on the
frequency grid, its pseudo-inverse is constant and the per-window cost is one
matrix multiply: 0.002 ms versus 0.623 ms for an iterative fit, agreeing to
8.8e-15. This retains excitation/inhibition balance as a real-time-viable
feature. It is a **surrogate** for a SpecParam/FOOOF exponent — biased by
periodic peaks in the fit range, and assuming no spectral knee — and is labelled
as such.

Two features are computed in bands outside the passband. `dwt_cD1` covers
64–128 Hz and `dwt_cD2` covers 32–64 Hz, both largely in the filter's stopband.
Dropping all 36 such features (291 → 255) retained **90–99% of within-subject
lift on four of five held-out records**, confirming the result does not depend
on stopband content.

### 2.6 Evaluation

We report AUPRC **against its baseline** (the positive rate), event-level
sensitivity at a fixed false-alarm rate per hour, **alarm duty cycle**, and
detection latency from annotated onset under a 3-of-5 persistence rule. The
persistence rule imposes an architectural latency floor of
`window + (k−1)·stride` = **4 s**.

Alarm duty is reported alongside false-alarm rate because refractory merging
collapses continuous alarming into a small number of "events". In validation, a
random model scored event sensitivity 1.00 at 2.02 FA/h with duty 0.50 — a
detector that is simply always on, which FA/h alone does not reveal.

---

## 3. Results

### 3.1 Cross-subject performance is unevenly distributed

Leave-one-subject-out logistic regression across 23 subject groups gives a
**median AUPRC lift of 28.6×** (mean 52.6× — the distribution is right-skewed
and the median is the appropriate summary). Ten of 23 folds reach event
sensitivity ≥ 0.8.

**Six of 23 folds sit at chance**: chb06 (2.8×), chb12 (1.3×), chb13 (1.1×),
chb14 (2.3×), chb15 (1.7×), chb16 (3.6×), all with 0.00–0.05 event sensitivity.
No intermediate cases were observed in this protocol.

We emphasise what this does *not* say. Seventeen of 23 subjects transfer
successfully, consistent with published continuous-data cross-subject results
(e.g. 75.34% event sensitivity under leave-one-out with a random forest). The
finding is **per-subject heterogeneity**, not universal cross-subject failure,
and a pooled sensitivity would conceal it in either direction.

Alarm duty remained 0.001–0.004 across every fold, so the failing detectors were
not simply always-on.

### 3.2 Evidence for patient-specific discriminative structure

**Experiment 1 — capacity.** A random forest (100 trees) on the four worst folds
gives 1.7×, 2.4×, 0.8× and 0.9× — no improvement, and two below chance. The same
model reaches 156× on chb01 with identical features and training data. This
additionally excludes decision-boundary geometry: random forests (axis-aligned
partitions) and logistic regression (a global hyperplane) have opposite
inductive biases and fail identically.

**Experiment 2 — feature adequacy.** Patient-specific training on chb15,
holding out whole records, gives **22.0×, 26.2×, 56.2×, 77.7× and 107.3×**
(median 56×) against a cross-subject 0.9×. This excludes three explanations at
once: the features capture this subject's seizures, the labels are correct, and
capacity is sufficient. It also establishes a **within-subject ceiling** against
which recovery can be measured.

**Experiment 3 — amplitude mismatch.** Within-record z-scoring (median-centred,
so ictal windows do not shift the baseline) moves the four hard folds to 1.3×,
1.3×, 1.5× and 2.1×. Against chb15's 56× ceiling this recovers approximately
**4% of the gap**.

**Experiment 4 — direction.** Comparing coefficient vectors of a chb15-only and
a population logistic regression: **correlation *r* = 0.106, with 3 of 20 top
features in common**. Five of chb15's eight highest-weighted features are the
complete left parasagittal chain (FP1-F3, F3-C3, C3-P3, P3-O1) — a plausible
lateralized focus that a population model cannot represent.

Per-feature normalization is a diagonal affine correction and cannot repair a
rotation of the discriminative direction, which is consistent with Experiment 3
recovering so little.

**This mechanism is supported but not uniquely identified.** Coefficient
correlation is a coarse instrument: the two models were standardized against
different statistics, and a low correlation is compatible with several
underlying causes. Covariance-based alignment, adversarial domain adaptation and
tangent-space transfer all remain untested here, and any of them succeeding
would refine or overturn this interpretation.

### 3.3 The patient calibration curve

We train a population model on the other 22 subject groups, then add the
patient's first *n* seizure-bearing records to training (sample-weighted 5×),
reserving the last 40% of that patient's seizure records as a **fixed test set**
that is never touched. Detection thresholds are selected on the *remaining*
calibration records, never on test data.

Event sensitivity, with quantile-transferred thresholds at a 2.0 FA/h target:

| subject | *n*=0 | *n*=1 | *n*=2 | *n*=3 | *n*=5 |
|---|---|---|---|---|---|
| chb15 | 0.00 | 0.17 | 0.17 | 0.75 | **0.92** |
| chb12 | 0.00 | 0.16 | 0.21 | 0.47 | **0.68** |
| chb13 | 0.20 | 0.40 | 0.20 | **0.60** | — |
| chb06 | 0.00 | 0.67 | 0.67 | 0.00* | — |
| chb09 *(control)* | **1.00** | — | — | — | — |
| chb10 *(control)* | **1.00** | 1.00 | 1.00 | 1.00 | — |

\* calibration-data exhaustion; see §5.

False-alarm rate remained within 0.0–3.8 h⁻¹ on every row. AUPRC lift over the
same range: chb15 1.5 → 9.3×, chb12 1.3 → 7.9×, chb13 2.0 → 12.6×, chb06
3.4 → 21.9×; controls 55–70× and flat.

**Patient calibration materially improves several initially hard subjects**,
with recovery varying by subject and by the number of calibration records
available. Every hard subject begins at chance and rises; both controls begin at
55–59× with sensitivity 1.00 and remain there. There is no overlap between the
groups at *n*=0. Controls do gain in ranking (chb10: 54.9 → 69.7×) while
sensitivity stays pinned at ceiling — patient data helps everyone, but only
*matters* for the subjects who need it.

An important methodological note: an earlier design in which the test set shrank
as training data grew **inflated results approximately threefold** (chb15 at
*n*=1: 7.9× with a shrinking test set versus 2.6× with a fixed one). The records
consumed for training were the chronologically earliest, and the remainder
differed systematically.

### 3.4 Two threshold-transfer findings

**Absolute thresholds do not transfer across sessions.** Selecting a threshold
on calibration records and applying that numeric value to test records fails:
chb12 passed calibration at duty ≤ 0.02 and then alarmed **25.9% of the time**
on test — same model, same threshold, a tenfold difference in duty. Per-record
score distributions shift with session amplitude and noise.

Transferring the *quantile* instead resolves it:

| | median test duty | maximum test duty |
|---|---|---|
| absolute threshold | 0.0096 | **0.0903** |
| quantile transfer | 0.0028 | **0.0071** |

False-alarm rate across all runs went from a 0.0–22.9 h⁻¹ range to 0.0–3.8 h⁻¹.
This is also the *deployable* form: a device maintains a running score
distribution and alarms at a fixed percentile of its own recent history, which
is causal and adapts to session drift.

**Duty monitoring is required.** The pathological chb12 run scored event
sensitivity 1.00 at 22.9 FA/h. Refractory merging made an always-on detector
appear merely noisy; alarm duty of 0.259 identified it immediately.

Threshold selection should additionally be constrained on **duty first,
false-alarm rate second**: FA/h estimated from one calibration record derives
from a handful of merged events and is high-variance, while duty derives from
thousands of windows.

### 3.5 A deployable patient-specific decoder

We train a causal temporal convolutional network on feature sequences — 291
features per timestep, one timestep per 2 s window. Four residual blocks,
kernel 3, 32 hidden channels, dilations 1/2/4/8, giving a **61-timestep (61 s)
receptive field**.

Causality is verified rather than assumed. Perturbing a single timestep by +100
changes no earlier output (max |Δ| = 0.000e+00), and scoring a prefix reproduces
the corresponding slice of a full forward pass exactly. A non-causal TCN trains
normally and is undeployable, and no loss curve reveals it.

**Population pretraining exhibits negative transfer.** A model pretrained on 22
subjects (342,519 windows, 178 events) scores **AUPRC lift 0.79× and AUROC
0.411 on chb15 — below chance**, ranking ictal windows lower than interictal.
Negating its scores gives 4.35×, indicating the learned representation is
systematically *inverted* for this subject rather than uninformative. Fine-tuning
a 33-parameter head cannot repair this (0.72×, declining). Full fine-tuning
recovers to 14.81×, but an identical architecture from **random initialization
reaches 15.57× and AUROC 0.992 versus 0.939** — and scores 5.98× at epoch zero,
before any gradient step, exceeding the pretrained model's entire population
training. Per-subject models are therefore trained from random initialization.

Across six subjects (medians over three random seeds; sensitivity σ = 0 on every
subject):

| subject | AUPRC lift | sens | FA h⁻¹ | latency | logreg sens @ FA h⁻¹ |
|---|---|---|---|---|---|
| chb06 | 538.6× | **1.00** | 1.60 | **5 s** | 0.67 @ 3.67 |
| chb09 *(ctrl)* | 114.5× | 1.00 | 0.54 | **5 s** | 1.00 @ 1.11 |
| chb10 *(ctrl)* | 92.3× | 1.00 | 1.04 | **6 s** | 1.00 @ 1.03 |
| chb12 | 21.2× | **0.91** | 0.41 | 13.5 s | 0.68 @ 1.67 |
| chb13 | 17.7× | 0.60 | 1.49 | 18 s | 0.60 @ 0.73 |
| chb15 | 14.8× | 0.86 | 0.40 | 21 s | 0.92 @ 3.02 |

**The TCN generally reduces false-alarm rate and latency; sensitivity is equal
or better on five subjects and lower on chb15** (0.86 vs 0.92, with a 7.5-fold
reduction in false alarms on that subject). It gives lower FA/h on five of six
and lower latency on all six. Four of six detect within 6 s against a 4 s
architectural floor.

These figures are **competitive with but do not exceed** published
patient-specific CHB-MIT results. Reported values commonly fall in the 95–100%
sensitivity, 0.08–0.66 FA/h, 2–8 s delay range, though protocols, subject
subsets, annotation handling and post-processing vary substantially and
harmonization would be required for a direct comparison. A recent
channel-selection TCN reports 97.57% sensitivity at 0.11 FA/h with 6.91 s delay.
We therefore do not claim state-of-the-art detection; the contribution is in
§3.6.

### 3.6 Streaming fixed-point inference

Re-evaluating a 192-step window for each new timestep performs 192× the
necessary work. Caching dilated activations reduces this to one output column
per input:

| | MACs per window | activation cache |
|---|---|---|
| naive recompute | 11,286,528 | — |
| streaming | **58,784** | 2,953 values (5.8 KB int16) |

The streaming path reproduces the batch forward pass to 1.53e-06 on real EEG
features, and state reset is exact (0.000e+00).

Weights are quantized to int16 at **per-layer power-of-two scales**. The binding
constraint is the accumulator, not stored precision: precision-optimal shifts
(17–18) place the worst-case accumulator at 33–36 bits, overflowing int32 on
every layer. Reducing shifts to 12–13 costs ~1e-03 relative weight error and
brings the worst case to ~29.5 bits. GELU — the only transcendental in the
network — is evaluated by a 512-point lookup table with linear interpolation
(9.77e-05 maximum error).

**Quantization costs zero detection performance.** Scoring the full held-out set
(9,783 windows, 7 events) at matched quantiles:

| metric | float64 | Q15 | Δ |
|---|---|---|---|
| AUPRC lift | 15.5997 | 15.6019 | +0.0022 |
| sensitivity @ *q*=0.95 | 0.8571 | 0.8571 | **0.0000** |
| FA h⁻¹ @ *q*=0.95 | 2.3501 | 2.3501 | **0.0000** |
| alarm duty @ *q*=0.95 | 0.0015 | 0.0015 | **0.0000** |
| latency @ *q*=0.98 | 24.0 s | 22.5 s | −1.5 s |

Sensitivity, false-alarm rate and alarm duty are identical at every operating
point. The single difference is 1.5 s of median latency in favour of the
quantized model, arising from one borderline window crossing threshold earlier.
Maximum score error is 0.0158 against a range of 11.2 (0.14%); rank correlation
is 0.999999.

**Deployment footprint: ~82 KB** — 57.9 KB weights, 5.8 KB activation cache,
18.4 KB feature ring buffer. This fits in flash on any Cortex-M4 and in SRAM on
an M7. For comparison, a 100-tree random forest requires ~340 KB and does not
fit; logistic regression requires 1.2 KB.

---

### 3.7 Position relative to published results

| setting | representative literature | this work |
|---|---|---|
| Patient-specific, full montage | ~95–100% event sensitivity, 0.08–0.66 FA/h, 2–8 s delay (protocols and subsets vary substantially) | 0.60–1.00 sensitivity, 0.4–1.6 FA/h, 5–21 s latency across six subjects |
| Patient-specific, reduced-channel | 99.62% sensitivity, 0.22 FA/h, 3.3 s mean latency on one clinically selected channel across 13 selected cases, with re-annotated intervals | Same clinical premise (personalization, focal spatial structure) demonstrated with a fixed 18-channel decoder and a fixed-point deployment path |
| Cross-subject | 75.34% event sensitivity under leave-one-out with continuous data, emphasizing imbalance and leakage | High median AUPRC lift with six of 23 subjects near chance; per-subject heterogeneity rather than uniform failure |
| Domain adaptation | 86.4% cross-patient sensitivity at 0.08 h⁻¹ via adversarial alignment (requires protocol audit) | Identifies the shift these methods target; alignment methods untested here |
| Embedded inference | Operation budgets, stream-equivalence tests, quantized validation and memory footprints are rarely reported | 58,784 MACs/timestep, ~82 KB, verified causal state, Q15 equivalence |

**We are not leading on detection performance.** The differentiator is the
implementation validation in the final row, and the per-subject analysis in
§3.1–3.2.

## 4. Discussion

The practical implication is that **cold-start cross-subject detection is
unreliable in a patient-dependent way**, which is a problem for a device even
when it succeeds on average. It failed for a well-defined subset of patients
here, and neither the tested random forest nor the population-pretrained TCN
remedied it under the evaluated protocol. Domain adaptation and
representation-alignment methods remain untested, so we do not claim the failure
is irreducible — only that it was not reduced by increased capacity or by
population data. This is consistent with why deployed responsive
neurostimulation systems are programmed per patient.

The useful question is instead the calibration curve: how much of a patient's
own data is needed, and what performance does each increment buy. That curve is
a device specification — it answers what a clinician asks — and it is reported
here for six subjects.

The negative transfer result is worth emphasizing because it is stronger than a
null. A population-pretrained network is not merely uninformative for a failing
subject; its learned representation is *inverted*, and a randomly initialized
model of identical architecture outperforms it before any training. This
predicts that transfer-learning approaches to cross-subject EEG will
underperform per-subject training on the affected subjects, not merely fail to
help.

We have not tested covariance-based alignment. Riemannian tangent-space methods
recenter each subject's covariance distribution to a common geometric mean,
addressing a *rotation* of the discriminative direction rather than a shift or
scale, and they operate on unlabelled target data. Adversarial domain alignment
has been reported on CHB-MIT with cross-patient sensitivity of 86.4% at
0.08 h⁻¹, though under a protocol that would require auditing before comparison.
Both are natural next tests, and either succeeding would revise §3.2's
interpretation.

---

## 5. Limitations

**The TCN result covers six subjects, not 23.** Three of the six (chb06, chb09,
chb10) have only 3–4 test events, so a sensitivity of 1.00 carries a 95%
confidence interval of approximately [0.44, 1.00]. chb12 (11 events) and chb15
(7 events) carry the argument. Extension to the full corpus is required.

**One subject does not benefit.** chb13 reaches AUROC 0.763 against 0.936–0.999
elsewhere, and its event sensitivity of 0.60 matches logistic regression
exactly.

**Calibration and training data compete.** In the calibration-curve protocol,
records used for training are unavailable for threshold calibration. At *n*=3,
chb06 calibrates on a single record, selects an unrepresentative quantile, and
reports 0.00 sensitivity despite achieving the **highest AUPRC lift of its
series** (21.9×). A dedicated calibration hold-out would remove this.

**Analyses were not preregistered.** All results reported here are
retrospective. The OSF preregistration commitment applies to subsequent
outcome analyses.

**Single corpus, pediatric population.** CHB-MIT is 23 pediatric subjects from
one institution. The bimodality reported here may not generalize to adult
scalp EEG or to intracranial recordings.

**Fixed-point validation is simulated.** The Q15 path is validated in NumPy
against float64, not on target silicon. Timing figures (0.175 ms per timestep)
are from interpreted code on a workstation core.

**Detection performance does not lead the literature.** Patient-specific CHB-MIT
results commonly reach 95–100% sensitivity at 0.08–0.66 FA/h with 2–8 s delay.
Our six-subject figures fall below that range on all three axes. We have not
attempted channel selection, re-annotation, or the post-processing used in
several of those reports, any of which could close part of the gap.

**Comparisons are not protocol-harmonized.** Published results differ in subject
subsets, segment- versus event-level scoring, annotation revision, train/test
splitting, and whether evaluation uses continuous recordings or balanced
extracts. The ranges quoted here are context, not benchmarks.

---

## 6. Conclusion

Cross-subject seizure detection on CHB-MIT succeeds for most subjects and fails
near-completely for an identifiable minority, with no intermediate cases in this
protocol. Model capacity, feature adequacy and amplitude mismatch were each
excluded as explanations; the pattern is consistent with patient-specific
discriminative structure, though alignment-based methods that could test this
directly remain unexamined. Patient calibration materially improves the affected
subjects and leaves the others at ceiling.

The resulting detector — patient-specific, causal, and quantized to Q15 —
reaches 0.60–1.00 event sensitivity at 0.4–1.6 false alarms per hour with
5–21 s latency in 58,784 MACs per timestep and ~82 KB. These figures are
competitive with, but do not exceed, published patient-specific results. The
contribution is that the implementation path is **validated end to end**:
bit-identical causal stream equivalence, an explicit operation and memory
budget, and fixed-point quantization measured to cost zero detection
performance at every tested operating point — a combination not reported, to our
knowledge, in prior CHB-MIT work.

---

## Data and code availability

CHB-MIT is publicly available from PhysioNet (doi:10.13026/C2K01R). All analysis
code, the decision log, and derived figures are available at [repository URL].
Derived outputs follow BIDS derivative conventions.

## Acknowledgements

[To be completed.]

---

## References

Andrzejak, R. G., et al. (2001). Indications of nonlinear deterministic and
finite-dimensional structures in time series of brain electrical activity.
*Physical Review E*, 64(6), 061907.

Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan,
J. W. (2010). A theory of learning from different domains. *Machine Learning*,
79(1–2), 151–175.

Esteller, R., Echauz, J., Tcheng, T., Litt, B., & Pless, B. (2001). Line length:
an efficient feature for seizure onset detection. *Proc. IEEE EMBS*.

Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic
excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78.

Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation*, 101(23), e215–e220.

Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (v1.0.0). PhysioNet.

Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment* (PhD thesis). MIT.

Wald, A. (1947). *Sequential Analysis*. Wiley.

[Additional references to be added: Halford et al. on inter-rater reliability;
Jacobs et al. on HFOs; Perucca et al. on onset patterns; Barachant et al. on
Riemannian classification.]
