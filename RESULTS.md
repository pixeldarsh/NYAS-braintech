# Uncertainty-aware EEG sleep staging + LIFU acoustic modelling: results

**Status of evidence.** Everything below is computational. The EEG results come from 5 recordings of one public dataset (Sleep-EDF). The acoustic results come from a simplified 2D virtual head. Nothing here is experimentally validated, clinically validated, or involves ultrasound delivered to a human or animal. Numbers describe model behaviour, not physiological outcomes.

## 1. EEG sleep-stage classification

Setup: Sleep-EDF, 5 participants, 13,640 30-s epochs (Wake 9297, N2 2462, REM 863, N3 530, N1 488), 10 Welch band-power features, leave-one-participant-out (LOSO).

### 1.1 Headline model and baselines (`baselines.py`)

| Model | Accuracy | Macro-F1 | N1 F1 | Err.-detection AUC (confidence) |
|---|---|---|---|---|
| Majority class (always Wake) | 0.682 | 0.162 | 0.00 | 0.50 |
| Stratified random | 0.510 | 0.199 | 0.03 | 0.50 |
| Logistic regression, all features | 0.732 | **0.591** | 0.21 | 0.78 |
| Logistic regression, relative power only | 0.690 | 0.535 | 0.22 | 0.58 |
| Random forest, relative power only | **0.768** | 0.557 | 0.24 | 0.81 |
| Random forest, all features | 0.751 | 0.561 | 0.23 | **0.86** |
| Original RF (`trainmodel.py`, saved predictions) | 0.765 | 0.551 | n/a (N1 recall 0.27) | 0.81 |

Findings
- Accuracy is dominated by Wake (68.2% of epochs); always predicting Wake gives 68%. Macro-F1 and per-stage F1 are the meaningful metrics.
- The random forest does **not** beat a regularised linear model on macro-F1 (LR 0.591 vs RF 0.551-0.561). The RF is better on accuracy and on how well its own confidence separates correct from incorrect epochs (AUC 0.81-0.86 vs 0.78).
- N1 is poorly classified by every model (F1 0.21-0.24), so it is a limit of band-power features at this sample size, not of one algorithm.
- Per-subject accuracy ranges 0.55-0.89 (original RF: SC4011E0 is hardest at 0.553). With n=5 subjects, differences of a few points between models are within between-subject variation and should not be called significant.
- The error-detection AUC is sensitive to the RF's randomness and feature ordering: the original run gave 0.79-0.82 (confidence 0.815, margin 0.815, entropy 0.795), the reproduction here 0.86. Report a range (about 0.79-0.86), not a single value.

### 1.2 Robustness to additive Gaussian noise (`robustness_compare.py`)

White noise added to raw epochs at fixed SNR, features recomputed, models trained on clean data. Pooled LOSO, mean of 3 noise seeds.

| Model | Clean F1 | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| RF (all) | 0.562 | 0.558 | 0.383 | 0.263 | 0.214 |
| RF (relative only) | 0.557 | 0.520 | 0.397 | 0.267 | 0.158 |
| LR (all) | 0.591 | 0.647 | 0.360 | 0.249 | 0.234 |
| LR (relative only) | 0.535 | 0.601 | 0.354 | 0.205 | 0.146 |

Calibration (expected calibration error) and confidence under noise, RF (all): ECE 0.04 clean, 0.15 at 5 dB, 0.17 at 0 dB, while mean confidence rises from 0.79 to 0.85. Error-detection AUC falls from 0.86 to 0.66.

Findings
- All models lose most macro-F1 between 20 and 10 dB; none of them is robust at 5 dB or below.
- The models become **more** confident as noise rises (RF mean confidence 0.79 to 0.85; LR 0.82 to 0.91) while becoming less accurate. This is the main reliability failure: confidence is not a safe indicator of quality under signal degradation. The RF's ability to flag its own errors drops from AUC 0.86 to 0.66 at 0 dB.
- Relative-power features are not uniformly more robust: they help the RF's error detection at low SNR (AUC 0.81 at 0 dB) but cost accuracy.
- Accuracy and macro-F1 at 20 dB are higher than clean for several models. The cause was not investigated (a shift in the noisy feature distribution that happens to favour the majority class is one candidate); it is not evidence that noise helps, and should be reported as an unexplained non-monotonic result.
- White Gaussian noise is a simple, optimistic stress test. It is not a model of real artefacts (movement, electrode pop, line noise).

### 1.3 Conformal prediction: does a coverage guarantee survive between-participant shift? (`conformal_eeg.py`)

Split conformal prediction, target coverage 90%. For each test participant, a different single participant calibrates the threshold and the other three train the model (all 4 calibration choices averaged). Score = 1 - probability of the true class.

| | LR | RF |
|---|---|---|
| Marginal coverage (target 0.90) | 0.836 | 0.851 |
| Mean prediction-set size | 1.72 | 1.43 |
| Share of epochs with a single-label set | 0.55 | 0.68 |
| Accuracy on single-label epochs | 0.82 | 0.87 |
| Coverage of N1 epochs | 0.74 | 0.57 |
| Coverage of REM / N3 epochs | 0.86 / 0.91 | 0.79 / 0.82 |
| Coverage, class-conditional (Mondrian) variant | 0.78 | 0.76 |
| N1 coverage, Mondrian | 0.83 | 0.88 |
| Mean set size, Mondrian | 1.48 | 1.59 |

Coverage by held-out participant (RF / LR): SC4001E0 0.95 / 0.96, SC4031E0 0.93 / 0.93, SC4041E0 0.86 / 0.87, SC4021E0 0.78 / 0.74, SC4011E0 0.73 / 0.68. Worst single test/calibration pair: 0.53 (RF), 0.50 (LR).

Findings
- The nominal 90% guarantee does not hold across participants (about 84-85% overall, 68-73% on the hardest participant). Conformal prediction assumes the calibration and test data are exchangeable; a different person's recording breaks that, so the shortfall is itself a measurement of participant-to-participant shift. Calibrating on a few more participants, or on a small amount of the test participant's own data, would be the obvious remedy and was not tested.
- Marginal coverage hides class failures: N1 is covered 57% (RF) against 90% nominal. Class-conditional calibration raises N1 coverage to 83-88% but lowers overall coverage (0.76-0.78) and does not restore the nominal level for any class under shift. The overall drop is probably because Wake, the majority class, gets a tighter class-specific threshold; this was not checked.
- Set size is a useful per-epoch uncertainty signal: single-label epochs (68% of epochs for RF) are right 87% of the time. In a two-step scheme, "accept singleton sets, defer the rest" gives a usable accuracy/coverage trade-off, but N1 would still be mostly deferred or missed.
- Calibration set is one participant (about 2-3k epochs, about 100 N1 epochs), so the class-conditional thresholds for N1 are noisy.

### 1.4 Blind test on 3 unseen participants (`blind_test.py`)

The model was frozen before scoring: original random forest settings, fitted on all 5 development participants, saved as `frozen_rf.joblib`, with no retraining or tuning afterwards. The test participants are SC4051, SC4061 and SC4071, the next three in numerical order (checksums verified against PhysioNet). They come from the same Sleep-EDF study, so this tests new people, not a new site, device or population.

| Evaluation | Model | Accuracy | Macro-F1 | N1 F1 |
|---|---|---|---|---|
| All epochs (primary) | Random forest | 0.798 | 0.610 | 0.16 |
| All epochs | Logistic regression | 0.755 | 0.599 | 0.14 |
| All epochs | Always Wake | 0.757 | 0.172 | 0.00 |
| Within 30 min of sleep | Random forest | 0.753 | 0.678 | 0.34 |
| Within 30 min of sleep | Logistic regression | 0.763 | 0.695 | 0.36 |
| Within 30 min of sleep | Always Wake | 0.190 | 0.064 | 0.00 |

| Participant | RF accuracy | Always-Wake accuracy | RF macro-F1 | RF accuracy, within 30 min of sleep | RF macro-F1, within 30 min of sleep |
|---|---|---|---|---|---|
| SC4051E0 | 0.750 | 0.829 | 0.480 | 0.690 | 0.584 |
| SC4061E0 | 0.912 | 0.747 | 0.727 | 0.771 | 0.685 |
| SC4071E0 | 0.733 | 0.697 | 0.628 | 0.780 | 0.727 |

Other blind results for the random forest: error-detection AUC 0.854; 90% conformal coverage 0.939 with mean set size 1.53, using a threshold from all 5 development participants' out-of-fold predictions (N1 coverage 0.78).

Gating, pooled over the 3 participants:

| Target | Policy | Triggers | Precision | Recall | Triggers in true Wake |
|---|---|---|---|---|---|
| N2 | argmax | 1459 | 0.65 | 0.92 | 19.1% |
| N2 | confidence >= 0.7 | 936 | 0.88 | 0.80 | 0.7% |
| N2 | confidence >= 0.7, 3 consecutive | 605 | 0.94 | 0.55 | 0.2% |
| N3 | argmax | 347 | 0.87 | 0.70 | 6.9% |
| N3 | confidence >= 0.7 | 232 | 0.96 | 0.51 | 0% |
| N3 | confidence >= 0.7, 3 consecutive | 128 | 1.00 | 0.30 | 0% |

Findings
- Performance held up on new people: macro-F1 0.61 and accuracy 0.80, against 0.55 and 0.77 in development. This is not evidence of real improvement. Three participants can differ by chance, and the frozen model also trained on 5 people rather than 4.
- Accuracy over all epochs is barely above always predicting Wake (0.80 vs 0.76), and for SC4051E0 it is below it (0.75 vs 0.83). Restricting to within 30 min of sleep drops the Wake baseline to 0.19, which confirms that long Wake periods inflate all-epoch accuracy.
- Logistic regression and the random forest remain close. The random forest has higher all-epoch accuracy; logistic regression is slightly higher near sleep. N1 remains the weakest stage.
- Conformal coverage met the target (0.94), unlike the development test with one calibration participant (0.85). The calibration sets differ, so this suggests more calibration people help but does not prove it.
- N3 gating did not fail here as it did in development: precision 0.87 vs 0.27, and 7% vs 60% of triggers in Wake. The development result (section 3.2) was therefore too pessimistic, or the true value varies widely between people; these data cannot separate the two.
- Unfiltered N2 gating is worse than in development (precision 0.65 vs 0.77). For SC4051E0, 42% of its 493 triggers were correct and 42% fell in true Wake. Filtering raised that participant's precision to 0.90.

These participants have now been seen. Using these results to change the model would mean they no longer count as blind.

### 1.5 Wake vs sleep on the blind participants (`wake_sleep.py`)

Evaluation of the frozen 5-stage model with no retraining: P(sleep) = P(N1) + P(N2) + P(N3) + P(REM). AUC is 0.980 over all epochs and 0.965 within 30 min of sleep (development: 0.937 and 0.888).

| Cut-off | Scope | Accuracy | Always-majority | Balanced accuracy | Sleep detected | Wake detected | Kappa |
|---|---|---|---|---|---|---|---|
| P(sleep) >= 0.50 | All epochs | 0.788 | 0.757 | 0.855 | 0.985 | 0.724 | 0.55 |
| P(sleep) >= 0.50 | Within 30 min of sleep | 0.883 | 0.810 | 0.716 | 0.985 | 0.447 | 0.53 |
| P(sleep) >= 0.73 | All epochs | 0.898 | 0.757 | 0.913 | 0.943 | 0.883 | 0.75 |
| P(sleep) >= 0.73 | Within 30 min of sleep | 0.918 | 0.810 | 0.878 | 0.943 | 0.812 | 0.74 |

The 0.73 cut-off maximised balanced accuracy on development out-of-fold predictions and was applied unchanged. The decision to look for a better cut-off was made after seeing the 0.50 result on the blind participants, so the 0.73 row is less strictly blind than the 0.50 row. Per-participant accuracy at 0.73, all epochs: 0.857, 0.964, 0.871; within 30 min of sleep: 0.891, 0.912, 0.942.

Stated confidence vs observed accuracy, within 30 min of sleep, using the 0.50 decision rule:

| Confidence | Share of epochs | Observed accuracy |
|---|---|---|
| 0.95-1.00 | 65% | 0.996 |
| 0.90-0.95 | 9% | 0.940 |
| 0.80-0.90 | 8% | 0.840 |
| 0.70-0.80 | 6% | 0.552 |
| 0.60-0.70 | 7% | 0.398 |
| 0.50-0.60 | 6% | 0.490 |

Findings
- Wake vs sleep is much easier than 5-stage staging, and the model ranks epochs well. The default 0.50 cut-off misses over half of Wake near sleep.
- Confidence is trustworthy at 0.80 and above. Below 0.80 it overstates accuracy, and those epochs (about 19% near sleep) should be treated as uncertain.
- At the 0.73 cut-off, N1 is the main source of error near sleep (0.693 correct), and 19% of Wake near sleep is called sleep.
- This confidence check is on clean signals only. Section 1.2 shows confidence becomes unreliable as noise increases.

## 2. Acoustic modelling (k-Wave Python, 2D)

Setup: 32 x 32 mm domain, elliptical head (scalp / skull / CSF / brain; skull 2800 m/s and 1900 kg/m^3), 1 MHz continuous sine, 1 mm source aperture at x = 4 mm, 20 us simulated, metrics over 5-20 us.

### 2.1 Numerical stability and the CFL failure (`cfl_failure_test.py`)

k-Wave's own stated limit for 2D pseudospectral is CFL = c_max dt / dx <= 0.45. Empirical test on the virtual head (dx = 0.25 mm, 10 us, only dt varied):

| CFL | Result |
|---|---|
| 0.1 - 0.6 | Stable (peak 0.356-0.369 Pa); warning issued above 0.45 |
| 0.8 | Unstable (peak 40 Pa) |
| 1.0 | Unstable (peak 3.9e13 Pa) |
| 1.5 | Unstable (peak 1.2e4 Pa) |

- The original 0.0625 mm run (dt = 20 ns, CFL = 0.896) is in the unstable region, which explains the failure.
- Correction to earlier advice: dt = 10 ns gives CFL = 0.448 at 0.0625 mm, which is just inside k-Wave's 0.45 limit, so it would likely have run. I used 5 ns (CFL 0.224) and a self-imposed guard of 0.3 for margin; both are conservative, not required.
- Peak pressure varies by about 3% across the stable CFL range (0.356 to 0.369 Pa), so time-step choice contributes a small additional error.
- Using one `dt` for all grids isolates spatial error from temporal error.

### 2.2 Grid convergence at the point (19, 16) mm (`grid.py`, dt = 5 ns)

| dx (mm) | Point RMS (Pa) | Peak (Pa) | RMS vs finest | Peak vs finest |
|---|---|---|---|---|
| 0.5 | 0.0686 | 0.1231 | -27.9% | -28.6% |
| 0.25 | 0.1149 | 0.2013 | +20.7% | +16.8% |
| 0.125 | 0.0998 | 0.1785 | +4.9% | +3.6% |
| 0.0625 | 0.0952 | 0.1724 | reference | reference |

- Successive changes shrink by about 3.2x, i.e. an observed order of roughly 1.7. A Richardson-type extrapolation puts the finest-grid error near 2% (limiting RMS about 0.093 Pa). This rests on three points and a staircased geometry, so it is an estimate only.
- 0.5 mm is under-resolved (about 3 points per wavelength in brain) and sits outside the converging regime. The earlier "substantial resolution dependence" came mostly from the 0.5 and 0.25 mm grids.

### 2.3 Region-averaged results and sensitivity (`sensitivity.py`, brain-region RMS, 5-20 us)

| Scenario | RMS at 0.25 mm | vs baseline | RMS at 0.125 mm | vs baseline |
|---|---|---|---|---|
| Homogeneous control | 0.1164 Pa | +32.9% | 0.1161 Pa | +38.9% |
| Virtual head (baseline) | 0.0876 Pa | - | 0.0836 Pa | - |
| Skull speed 2400 m/s | 0.0883 Pa | +0.8% | 0.0841 Pa | +0.5% |
| Skull speed 3200 m/s | 0.0756 Pa | -13.7% | 0.0771 Pa | -7.8% |
| Skull density 1700 | 0.0897 Pa | +2.4% | 0.0863 Pa | +3.2% |
| Skull density 2100 | 0.0856 Pa | -2.2% | 0.0811 Pa | -3.0% |

Finest-grid check (`sensitivity.py 0.125:2 0.0625:4`). Brain sensor subsampled onto a 0.25 mm lattice so the 512 x 512 run fits in memory; at 0.125 mm the subsampled RMS (0.083588 Pa) matched the full-region value (0.083606 Pa).

| Scenario | RMS at 0.125 mm | vs baseline | RMS at 0.0625 mm | vs baseline |
|---|---|---|---|---|
| Homogeneous control | 0.1161 Pa | +38.9% | 0.1160 Pa | +41.7% |
| Virtual head (baseline) | 0.0836 Pa | - | 0.0818 Pa | - |
| Skull speed 3200 m/s | 0.0770 Pa | -7.8% | 0.0777 Pa | -5.1% |

Baseline brain RMS across grids: 0.0876 (0.25 mm), 0.0836 (0.125), 0.0818 (0.0625). Successive changes shrink by about 2.3x (observed order about 1.2), so the extrapolated limit is about 0.0805 Pa and the finest-grid error is roughly 2%.

- Homogeneous-vs-layered difference: +33%, +39%, +42% from coarse to fine. The control is flat while the layered result slowly decreases, so the gap grows with resolution and is not yet converged. Quote it as "about 40% (+33% to +42% across grids)".
- Skull speed 3200 m/s effect: -13.7%, -7.8%, -5.1%. It is still shrinking at the finest grid, so the coarse-grid magnitude was an overestimate and the converged value is not established. The direction (lower brain RMS with faster skull) is consistent.

Findings
- Heterogeneity is the dominant effect: the homogeneous control reads 33-42% higher brain RMS than the layered head, and the control is grid-insensitive (0.1164, 0.1161, 0.1160 Pa), so the difference is geometry/material and not numerics.
- Skull density has a small, grid-consistent effect (about +/-2-3% RMS for +/-200 kg/m^3).
- Skull sound speed is asymmetric: -400 m/s changes RMS by under 1%, +400 m/s by -5% to -14% depending on grid (still shrinking at 0.0625 mm, so its true size is unresolved). Only the direction should be quoted.
- Peak pressure is not a reliable sensitivity metric here: scenario differences change sign between grids and the homogeneous peak is 25-30% below the layered peak. Use RMS.
- Sensitivities are one-at-a-time and local, with other properties fixed. They are not uncertainty intervals.

### 2.4 Monte Carlo propagation of tissue-property uncertainty (`uq_acoustic.py`, 0.125 mm, 24 Latin-hypercube samples)

Varied together: skull speed 2400-3200 m/s, skull density 1700-2100 kg/m^3, brain speed 1500-1580 m/s, brain density 1020-1060 kg/m^3. These ranges are assumed bounds, not fitted to measurements.

| Brain RMS pressure | Value |
|---|---|
| Mean | 0.0826 Pa |
| Standard deviation | 0.0029 Pa (CV 3.5%) |
| Min / median / max | 0.0773 / 0.0831 / 0.0885 Pa |
| 5-95% interval | 0.0776 - 0.0873 Pa |

Parameter importance (Spearman rank correlation with RMS; linear fit R^2 = 0.87):

| Parameter | Spearman rho | Standardised coefficient |
|---|---|---|
| Brain sound speed | +0.60 (p = 0.002) | +0.48 |
| Skull sound speed | -0.58 (p = 0.003) | -0.58 |
| Skull density | -0.46 (p = 0.024) | -0.43 |
| Brain density | +0.15 (p = 0.49) | +0.14 |

Findings
- Over these property ranges, brain RMS varies by about 3.5% (5-95% interval roughly +/-6% around the mean). That is small next to the structural effect of layering (about 40% vs a homogeneous medium) and comparable to the numerical error at 0.125 mm (about 2% in region RMS), so property uncertainty and discretisation error are of similar size and should be reported together.
- Brain sound speed matters about as much as skull sound speed; a sensitivity study that varies only skull properties omits a comparably important parameter. Brain density has no detectable effect.
- Direction: faster skull and denser skull lower brain RMS; faster brain raises it.
- Limits: 24 samples is small (rank correlations are rough, p-values indicative only); inputs are sampled independently and uniformly although real tissue properties are correlated; geometry (skull thickness, shape) was not varied, and geometric uncertainty could be larger than property uncertainty.

### 2.5 Focused-beam head model with absorption (`lifu_model.py`)

**Stopped early.** These runs were stopped to save time with 27 of 43 planned simulations finished. The 0.125 mm grid runs and the 3D benchmark comparison did not finish, and the Monte Carlo reached 19 of 32 samples. Finished simulations are in `lifu_cache/` and `lifu_finished_so_far.csv`.

Setup: 2D, 500 kHz, 64 mm radius-of-curvature arc source with a 64 mm aperture, curved skin / skull / brain layers 30 mm from the source. Tissue properties and layer thicknesses follow Aubry et al. 2022 (JASA 152:1003), Table I and benchmarks 4 and 6. That paper calls its values representative rather than definitive. A 2D arc gives a line focus, not a 3D bowl focus, and the far side of the skull lies outside the domain, so reflections from it are not modelled.

Independent checks completed:

| Check | Reference | Result |
|---|---|---|
| Transmission through bone layers, 1D plane wave | Exact transfer-matrix solution | Within 0.5% at 0.25 and 0.125 mm; a +0.44% offset for the absorbing plate does not shrink with grid refinement (cause not identified) |
| 2D focus in water, 0.25 mm | Numerically integrated analytic field | Axial 27.09 vs 27.12 mm, lateral 3.50 vs 3.51 mm, on-axis shape error 0.10% |
| Run length and precision, 0.25 mm | Same model | 140 us and float64 runs matched to 3 decimals; 80 us matched 100 us to 4 digits in peak pressure |

Results at 0.25 mm, relative to the same source in water at the same grid (`lifu_scenarios.csv`):

| Model | Focal pressure | Focus shift from target | Axial -6 dB | Lateral -6 dB |
|---|---|---|---|---|
| Water | 1.00 | 1.1 mm | 27.1 mm | 3.5 mm |
| Brain only, no skull | 0.91 | 2.3 mm | 26.6 mm | 3.6 mm |
| Single-layer skull, no absorption | 0.77 | 4.7 mm | 30.9 mm | 3.7 mm |
| Single-layer skull, absorption | 0.51 | 4.5 mm | 30.8 mm | 3.8 mm |
| 3-layer skull, no absorption | 0.54 | 1.3 mm | 28.6 mm | 3.7 mm |
| 3-layer skull, absorption | 0.34 | 2.3 mm | 29.0 mm | 3.8 mm |

Findings so far
- Absorption and skull structure both matter: the full model delivers 0.34 of the water focal pressure, and a single solid skull layer overestimates that by about 50% (0.51 vs 0.34).
- As a rough cross-check, plane-wave transmission through the same layers (0.405) times loss in the brain (about 0.91) gives about 0.37, close to the simulated 0.34.
- Water itself peaks 1.1 mm short of the geometric focus, so part of each reported shift is not caused by the head.
- These are simulated, uncalibrated values, not safety or dosing quantities.

Grid dependence, 3-layer skull with absorption (two grids only):

| dx | Focal pressure vs water | Focus position from source | Axial -6 dB | Lateral -6 dB |
|---|---|---|---|---|
| 0.5 mm | 0.337 | 61.74 mm | 28.6 mm | 3.70 mm |
| 0.25 mm | 0.339 | 61.67 mm | 29.0 mm | 3.76 mm |

The two grids agree closely (0.6% in relative focal pressure), but convergence is **not established**: the 0.125 mm runs did not finish, and two grids cannot show a trend. The ratio to water may also hide errors shared by both runs. The 1D and 2D analytic checks above give indirect support at 0.25 mm.

Monte Carlo at 0.25 mm (`lifu_mc.csv`), varied together: bone speed and density x0.9-1.1, bone absorption x0.5-2, skull thickness x0.75-1.25, brain speed 1513-1607 m/s. These are assumed spreads around published values. 19 of 32 samples finished, and none went unstable.

| Output | Mean | SD | 5-95% range |
|---|---|---|---|
| Focal pressure vs water | 0.371 | 0.101 (CV 27%) | 0.254-0.575 |
| Focus shift from target | 3.45 mm | 0.52 mm | 2.70-4.10 mm |
| Axial -6 dB | 29.8 mm | 1.33 mm | 28.1-31.9 mm |
| Lateral -6 dB | 3.76 mm | 0.09 mm | 3.65-3.94 mm |

| Input | Spearman rho with focal pressure | Standardised coefficient (linear fit R^2 = 0.89) |
|---|---|---|
| Bone absorption | -0.84 (p < 0.001) | -0.77 |
| Skull thickness | -0.57 (p = 0.011) | -0.44 |
| Bone speed | -0.28 (p = 0.24) | -0.08 |
| Bone density | -0.28 (p = 0.24) | -0.14 |
| Brain speed | +0.22 (p = 0.36) | -0.06 |

- Stopping early biased this sample. High-absorption samples use smaller time steps and run slowest, so the unfinished samples have more absorption (mean factor 1.43 vs 1.13 for finished; 1.25 planned). The finished mean of 0.371 therefore reads high. A log-linear fit to the finished samples, applied to all 32 planned inputs, gives a mean of about 0.35 and a 5-95% range of about 0.22-0.52. That is a model-based estimate, not a simulation result.
- Bone absorption and skull thickness dominate focal-pressure uncertainty. This uncertainty is far larger than in the earlier model without absorption or thickness variation (CV 27% vs 3.5%), and that comparison reverses the earlier finding that brain speed matters as much as skull speed: here brain speed has no clear effect.
- With 19 samples, correlations are rough and the p-values are not corrected for multiple comparisons.

Not completed: comparison with the published benchmark. The 3D bowl run comparing with Aubry et al.'s 26.2 mm axial and 4.1 mm lateral focal size never finished. The model has been checked against analytic solutions, not against that benchmark or any measurement.

New failure conditions found:
- Absorption tightens the stable time step as the grid gets finer. A fixed CFL of 0.2 went non-finite at 0.0625 mm in 1D, and at 0.125 mm in 2D. The model now also follows k-Wave's stability estimate with a 0.9 margin and stops with an error on non-finite pressure.
- k-Wave 0.6.2 returns sensor data in C order. Reshaping with `order="F"` scrambled an early version of these results, and an analytic check caught it. `acoustic.py` also uses F order for tissue-region selection and pressure snapshots, so its tissue-wise, brain-region and sensitivity values should not be used until that is corrected. `grid.py`, `sensitivity.py` and `uq_acoustic.py` are unaffected, and no earlier table in this report comes from `acoustic.py`.
- The bundled C++ k-Wave solver crashes on this machine, so all runs use the Python solver, and k-Wave's axisymmetric mode was not used.
- Memory limits this work on an 8 GB machine: running 3D and 2D jobs together caused heavy swapping and slowed everything, which is why the remaining runs were stopped.

## 3. EEG-acoustic integration: a software-only gating study (`integration.py`)

What this is. A hypothetical closed loop in which a sonication trigger would fire only when the EEG classifier predicts a chosen target stage. For each gating policy we measure how often triggers would be mistimed (the true stage is not the target), under clean and noisy EEG, and compare that variability with the variability from acoustic tissue-property uncertainty (Monte Carlo, section 2.4).

What this is not. It models no physiological effect of ultrasound, delivers nothing to anyone, and the two components share no physics: the classifier's output does not change the acoustic field, and the Sleep-EDF recordings have no anatomy that could set the head model. The target stage (N2 or N3 below) is an arbitrary example; nothing in this work supports targeting any stage. The only coupling is *timing reliability*: given an imperfect stage detector, how often is a trigger mistimed.

EEG classifier: the original random forest settings, leave-one-participant-out; noisy conditions score held-out epochs with white noise added to the raw signal (the model is trained on clean data). Epochs are in recording order within each participant; "3 consecutive" requires the trigger condition on 3 successive 30-s epochs.

### 3.1 Target = N2 (precision = share of triggers occurring in true N2; recall = share of N2 epochs triggered)

| EEG condition | Policy | Triggers | Precision | Recall | Triggers in true Wake | Worst / best participant precision |
|---|---|---|---|---|---|---|
| Clean | argmax = N2 | 2632 | 0.77 | 0.82 | 12.0% | 0.68 / 0.86 |
| Clean | + confidence >= 0.7 | 1845 | 0.91 | 0.68 | 0.8% | 0.89 / 0.92 |
| Clean | + confidence >= 0.9 | 1246 | 0.95 | 0.48 | 0.3% | 0.92 / 1.00 |
| Clean | + 3 consecutive epochs | 1667 | 0.91 | 0.62 | 0.4% | 0.81 / 0.95 |
| Clean | conf >= 0.7 and 3 consecutive | 1155 | 0.95 | 0.45 | 0.1% | 0.91 / 0.99 |
| 10 dB noise | argmax = N2 | 1222 | 0.90 | 0.45 | 4.0% | 0.79 / 0.98 |
| 10 dB noise | conf >= 0.7 and 3 consecutive | 169 | 1.00 | 0.07 | 0% | 1.00 / 1.00 |
| 5 dB noise | argmax = N2 | 662 | 0.83 | 0.22 | 7.1% | 0.21 / 1.00 |
| 5 dB noise | conf >= 0.7 | 63 | 0.95 | 0.02 | 0% | 0.00 / 1.00 |
| 5 dB noise | conf >= 0.9 | 0 | - | 0 | - | - |

### 3.2 Target = N3 (slow-wave sleep; the rarest and most imbalanced example)

| EEG condition | Policy | Triggers | Precision | Recall | Triggers in true Wake | Worst / best participant precision |
|---|---|---|---|---|---|---|
| Clean | argmax = N3 | 1104 | 0.27 | 0.57 | 60.5% | 0.09 / 1.00 |
| Clean | + confidence >= 0.7 | 442 | 0.43 | 0.36 | 37.8% | 0.19 / 0.98 |
| Clean | + confidence >= 0.9 | 139 | 0.50 | 0.13 | 8.6% | 0.46 / 1.00 |
| Clean | conf >= 0.7 and 3 consecutive | 111 | 0.59 | 0.12 | 17.1% | 0.41 / 1.00 |
| 10 dB, 5 dB noise | any policy | 0 | - | 0 | - | - |

### 3.3 Acoustic versus classification variability

- The acoustic delivery term, tissue-property uncertainty propagated to a relative brain-pressure index (RMS squared), has a coefficient of variation of 7.0% (3.5% in RMS).
- The across-participant variability of mistimed triggers is much larger: coefficient of variation 0.29 (N2, clean), 0.67 (N2, 10 dB), 1.33 (N2, 5 dB), 1.11 (N3, clean). Precision for the same policy ranges from 0.68 to 0.86 (N2, clean, argmax) and from 0.09 to 1.00 (N3).
- **Superseded in part by section 2.5.** Using the absorbing model with skull thickness varied, the same pressure-squared index has a coefficient of variation of about 0.57 (from 19 finished samples, with the bias noted there). That exceeds the clean N2 mistimed-trigger spread (0.29) and is below the noisy-EEG and N3 values (0.67-1.33), so neither source clearly dominates. The two measures also describe different things: one is spread across participants and the other is spread across assumed tissue properties.
- With the earlier model, the uncertainty in whether a trigger is correctly timed exceeded the acoustic property uncertainty by roughly 4 to 20 times. That ordering should be re-checked with geometric uncertainty (skull thickness), which was not varied.

### 3.4 Findings

- Gating on a well-represented stage (N2) can be made fairly precise in clean EEG: with a confidence threshold and persistence rule, about 95% of triggers fall in true N2 and almost none in Wake, at the cost of covering under half of N2 epochs. There is a direct precision/coverage trade-off; no policy gives both.
- Update from the blind test (section 1.4): this N3 finding did not replicate on 3 unseen participants (precision 0.87, 7% of triggers in Wake), so treat it as development-set evidence only.
- In the 5 development participants, gating on a rare stage (N3) was unreliable: the clean policies put 9-60% of triggers in true Wake (60% unfiltered, 9% at confidence >= 0.9, which then covers only 13% of N3 epochs), because Wake epochs outnumber N3 about 18 to 1 and the classifier confuses them. In these participants, a rare-stage trigger would mostly have fired at the wrong time.
- Noise destroys coverage rather than precision. Under noise the classifier almost never predicts the target (recall for N2 falls from 0.82 to 0.45 to 0.22; N3 triggers drop to zero), and the few remaining triggers look precise only because there are so few. Precision alone is therefore a misleading reliability measure; report it with recall and trigger count.
- A confidence threshold is not a safe filter under degradation: in section 1.2 mean confidence rises with noise while accuracy falls. Here the threshold happened to suppress triggers, but at 5 dB one participant's remaining triggers had 0% precision, so a confidence rule cannot stand in for a signal-quality check.
- Participant variability dominates: worst-participant precision is 0.68-0.81 against 0.86-0.95 best for N2 policies on clean data. Any claim about gating reliability needs per-participant reporting, and 5 participants is too few to state a population-level rate.
- A practical consequence for the framework: signal-quality gating (reject low-SNR epochs before staging) and a participant-specific calibration step are prerequisites, not refinements.

## 4. Failure conditions identified

| Area | Condition | Observed effect |
|---|---|---|
| EEG | Class imbalance (Wake 68%) | Accuracy overstates skill; always-Wake scores 68% |
| EEG | N1 epochs | F1 0.21-0.24 for every model tried |
| EEG | Held-out participant SC4011E0 | RF accuracy 0.553 vs 0.89 for the easiest participant |
| EEG | Additive noise <= 10 dB SNR | Macro-F1 drops from 0.56 to 0.38 (10 dB) and 0.21 (0 dB); confidence rises while accuracy falls |
| EEG | Reliance on confidence alone | Error-detection AUC 0.86 clean to 0.66 at 0 dB |
| Acoustic | CFL > 0.45 | Warning; blow-up by CFL 0.8 |
| Acoustic | dx = 0.5 mm (about 3 points/wavelength in brain) | -28% point RMS error; not usable |
| Acoustic | dx = 0.25 mm | +21% point RMS error vs finest; skull-speed (3200 m/s) effect overstated (-13.7% vs -5.1% at 0.0625 mm) |
| Acoustic | dx = 0.125 mm | About 5% point and 2% region RMS error; skull-speed effect still -7.8% vs -5.1% finest |
| Acoustic | Using peak instead of RMS | Inconsistent sign across grids |
| Acoustic | Absorption at fine grids or high absorption with fixed CFL | Non-finite pressure; needs a stability-limited time step |
| Acoustic | Single-layer skull instead of 3 layers | Focal pressure overestimated by about 50% (0.51 vs 0.34) |
| Integration | Gating on rare stage N3 | Development: 9-60% of triggers in true Wake (9% only at confidence >= 0.9, covering 13% of N3 epochs). Blind test: 0-7%, so this varies by participant |
| Integration | EEG noise <= 10 dB | Trigger coverage collapses (N2 recall 0.82 to 0.22 at 5 dB; N3 none); remaining precision unreliable |
| Integration | Participant differences | Per-participant precision 0.09-1.00 (N3), 0.68-0.86 (N2, clean, argmax) |

## 5. Limitations

- EEG: 5 development and 3 blind participants, 1 dataset, 1 channel, band-power features only; no confidence intervals beyond the per-participant spread; no hyperparameter search; no test on a different site or device.
- Acoustic, focused model (section 2.5): 2D line focus, idealised concentric layers, far skull not modelled, convergence and the published benchmark comparison not completed, assumed uncertainty ranges, no thermal or safety model, not calibrated.
- Acoustic, earlier models (sections 2.1-2.4): 2D, elliptical head with idealised layers, a constant-property skull, no attenuation or absorption, no thermal or safety model, staircased interfaces, one source geometry and one frequency, single-run (deterministic) simulations.
- The only EEG-acoustic coupling is the software gating study in section 3 (timing reliability of a stage detector). No result here links sleep-stage predictions to any acoustic field, physiological effect or anxiety-related outcome; the head model is not derived from the EEG participants.
- Simulated pressures (order 0.1 Pa for a 1 Pa source amplitude) are relative model quantities, not calibrated therapeutic or safety values.

## 6. Reproducing

Environment: Python 3.11, `.venv311` (k-wave-python 0.6.2, scikit-learn 1.9, numpy 2.2).

| Script | Output |
|---|---|
| `trainmodel.py` | `oof_predictions.csv` (original RF) |
| `baselines.py` | `baseline_comparison.csv`, `baseline_per_subject.csv` |
| `noise.py` | `noise_robustness_results.csv` (original RF) |
| `robustness_compare.py` | `robustness_comparison.csv`, `robustness_comparison_raw.csv` |
| `grid.py` | printed convergence table |
| `sensitivity.py [dx_mm ...]` | `sensitivity_results.csv`, `sensitivity_results_fine.csv` |
| `cfl_failure_test.py` | `cfl_failure_test.csv` |
| `conformal_eeg.py` | `conformal_results_raw.csv` |
| `uq_acoustic.py [dx_mm n_samples stride]` | `uq_acoustic_<dx>mm.csv` |
| `integration.py` | `integration_results.csv` |
| `blind_test.py` | `frozen_rf.joblib`, `blind_test_results.csv`, `blind_test_gating.csv` |
| `wake_sleep.py` | `wake_sleep_results.csv`, `wake_sleep_cutoff_results.csv` |
| `verify_acoustic.py plate\|arc2d\|bowl3d` | `verify_plate.csv`, `verify_arc2d.csv`, `verify_bowl3d.csv` |
| `lifu_experiments.py scenarios\|grid\|mc` | `lifu_scenarios.csv`, `lifu_mc.csv`, `lifu_cache/` (the `grid` mode did not finish, and `lifu_mc.csv` was built from the finished samples) |
| `lifu_collect.py` | `lifu_finished_so_far.csv` |
