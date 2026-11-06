# Nyas Braintech: when can we trust the tools?

A software-only research framework that tests how reliable two kinds of tools are for future sleep and anxiety research:

- **EEG sleep staging:** a classifier that labels 30-second EEG epochs as Wake, N1, N2, N3 or REM, with checks on its confidence, robustness to noise and performance on new people.
- **Low-intensity focused ultrasound (LIFU) simulation:** a 2D k-Wave model of a focused beam passing through skin, skull and brain, with checks against exact solutions and an analysis of how uncertain tissue properties change the result.

A small software test joins the two: how often a hypothetical sleep-stage trigger would fire at the wrong time.

> **Simulation and analysis only.** No ultrasound was given to any person or animal. Nothing here is experimentally or clinically validated, and simulated pressures are relative, uncalibrated values, not safety or dose values.

Built for the New York Academy of Sciences Junior Academy Innovation Challenge.

## Main results

Full results, methods, limits and failure conditions are in [RESULTS.md](RESULTS.md).

- **Blind test on 3 new people:** 79.8% accuracy and 0.61 macro-F1, but always guessing Wake scores 75.7%, so accuracy alone misleads. N1 is the weakest stage.
- **Confidence can fail:** as noise is added to the EEG, macro-F1 falls from 0.56 to 0.21 while mean confidence rises from 0.79 to 0.85.
- **Wake vs sleep:** about 92% accuracy within 30 minutes of sleep, using a cut-off taken from development data.
- **Acoustics:** the layered head leaves about one-third of the water focal pressure in this 2D model, and a single-layer skull overestimates it by about 50%. Bone absorption and skull thickness drive most of the uncertainty.

## Repository layout

| Path | Contents |
|---|---|
| `download_data.py`, `preprocesseeg.py`, `extractfeatures.py` | Download Sleep-EDF, cut 30-second epochs, compute Welch band-power features |
| `trainmodel.py`, `baselines.py` | Random forest with leave-one-participant-out testing, and simpler baselines |
| `noise.py`, `robustness_compare.py`, `errordetection.py`, `uncertainityanalysis.py`, `selectiveprediction.py`, `conformal_eeg.py` | Noise robustness, confidence, error detection and conformal prediction |
| `blind_test.py`, `wake_sleep.py` | Frozen-model test on 3 unseen participants, and wake-vs-sleep evaluation |
| `lifu_model.py`, `lifu_experiments.py`, `verify_acoustic.py`, `lifu_collect.py` | Focused-beam head model with absorption, its experiments and analytic checks |
| `grid.py`, `sensitivity.py`, `uq_acoustic.py`, `cfl_failure_test.py`, `acoustic.py` | Earlier 1 MHz acoustic model, grid convergence, sensitivity and stability tests |
| `integration.py` | Software-only trigger-timing study |
| `make_figures.py`, `make_deck_figures.py`, `make_beam_animation.py` | Figures and the beam animation |
| `*.csv`, `*.png`, `lifu_cache/`, `lifu_fields.npz`, `beam_animation.mp4` | Saved results, figures and simulation outputs |
| `Milestone_4_Presentation/` | Final presentation: PowerPoint, browser version, slide sources and media |
| `None-Junior_Academy_Milestone_3_-_FILLED.docx` | Milestone 3 results form |

## Reproducing

Python 3.11:

```bash
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
.venv311/bin/python download_data.py
.venv311/bin/python preprocesseeg.py
.venv311/bin/python extractfeatures.py
.venv311/bin/python trainmodel.py
```

Then run any analysis script. `blind_test.py` and `wake_sleep.py` expect the three blind recordings (SC4051, SC4061, SC4071) in `sleep_edf_blind/`. The fine-grid acoustic runs need several GB of memory and can take hours.

## Not included

- The Sleep-EDF recordings, `eeg_epochs.npy` and `frozen_rf.joblib`, which are large and can be rebuilt with the scripts above.
- NYAS's blank template and instructions documents.

## Known issues

- `acoustic.py` selects tissue regions with `order="F"`, but k-Wave Python 0.6.2 returns sensor data in C order, so its tissue-wise and brain-region values should not be used. No result in `RESULTS.md` depends on it.
- The 0.125 mm grid-convergence runs and the 3D comparison with the published benchmark were not completed, and the Monte Carlo study stopped at 19 of 32 samples.
- Bracketed placeholders (team names, local statistics, community feedback) still need to be filled in the presentation and forms.

## Data and references

EEG data: Sleep-EDF Database Expanded, from PhysioNet. Kemp B, et al. IEEE Trans Biomed Eng. 2000;47(9):1185–1194. Goldberger AL, et al. Circulation. 2000;101(23):e215–e220.

Acoustic model: k-Wave (Treeby BE, Cox BT. J Biomed Opt. 2010;15(2):021314). Tissue values and benchmark geometry: Aubry J-F, et al. J Acoust Soc Am. 2022;152(2):1003–1019.
