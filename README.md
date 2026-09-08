# Complaint severity triage model

A classification model that predicts the **Severity Score (1-6)** of a complaint about a public
service and, from it, the triage decision the score drives:

| Severity Score | Outcome |
|---|---|
| 1-3 | Not progressed |
| 4-6 | Progressed for further investigation |

The organisation has a low tolerance for severe cases being classified as low severity, so the
solution is built around that asymmetry rather than around raw accuracy.

## Deliverables

| Item | Where |
|---|---|
| Source code | `src/severity_triage/` (installable package with `severity-train` / `severity-predict` CLIs) |
| Notebook | `notebooks/severity_analysis.ipynb` (executed, with outputs and figures) |
| Holdback predictions | `predictions/holdback_predictions.csv` |
| README | this file |
| Requirements | `requirements.txt` (pinned to the versions used to produce the committed outputs) |
| Tests | `tests/` (18 tests, `make test`) |
| Trained model + metrics | `models/severity_model.joblib`, `models/metrics.json`, `models/model_card.json` |
| Data | `data/` (the original workbook plus CSV exports of both sheets) |

### Prediction file format

`predictions/holdback_predictions.csv`, one row per holdback case:

| Column | Meaning |
|---|---|
| `CaseReference` | case identifier |
| `PredictedSeverityScore` | predicted score 1-6 |
| `TriageDecision` | `Progress` / `Do not progress` (always consistent with the score) |
| `ProbabilitySevere` | model probability that the true score is 4-6 |
| `P_Severity1` … `P_Severity6` | full class probabilities |
| `TopDrivers` | the three features that pushed this case most towards or away from "severe" (SHAP) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -r requirements.txt && pip install -e .
make test             # unit tests
make train            # 5-fold CV + final fit -> models/
make predict          # score data/holdback_dataset.csv -> predictions/
make notebook         # re-execute the analysis notebook
```

`make all` runs everything (about three minutes on a laptop). All random seeds are fixed and LightGBM
runs single-threaded, so re-running produces the same model, metrics and predictions.

To score a different file: `severity-predict --input path/to/cases.csv --output out.csv --explain`.
A Dockerfile is included for a containerised scorer.

## Approach

### 1. Data preparation (`data.py`)

The data dictionary in the assessment workbook is treated as the contract and is encoded in
`config.py`. Cleaning is deterministic and identical for training and inference, and returns a
`DataQualityReport` for logging and monitoring. Rules, with the evidence behind each:

* **Leakage removed.** `OmbudsmanInvestigationRequired` is a perfect restatement of the target
  (`Yes` ⇔ score ≥ 4) and is empty in the holdback data, so it is an outcome of the decision, not an
  input. It is dropped and cannot reach the model.
* **Out-of-range values become missing.** Every numeric column contains exactly one value below and
  one above its dictionary range in each file (negative costs, a 0-100 score of 108, 3,650 days
  affected). They are treated as recording errors, set to missing and imputed, rather than clipped to
  a plausible-looking but invented value.
* **Structural "None".** `PhysicalImpactLevel`, `VulnerabilityLevel`, `PrimaryVulnerability` and
  `FailureTypeSecondary` list `None` as an allowed value, never contain it, and their blanks have the
  lowest mean severity of any level. Blanks are filled with `None`.
* **Other gaps** (1.5-6% in a dozen columns) are median-imputed inside the pipeline with a
  missing-indicator column, so imputation is learned on training folds only.
* **Jurisdiction** is a function of `OrganisationType` (NHS Trust / GP / ICB → Health; Government
  Department / Agency / Local Authority → Parliamentary) and is filled from it when blank.
* **Validation.** Unknown categories and unparseable booleans are counted and set to missing; missing
  required columns raise an error; unexpected columns are ignored with a warning.

### 2. Feature engineering (`features.py`)

* Ordered categoricals (impact levels, vulnerability, evidence strength, remedy band, age band,
  investigation route) are encoded as integers so the ordering is preserved.
* Nominal categoricals are one-hot encoded with unknown-safe handling.
* Engineered features follow the brief's own list of severity factors: total financial loss and log
  transforms of heavy-tailed money and duration fields; `HealthImpactCount` across the six health
  flags; `MaxImpactLevel` and a `VulnerabilityXImpact` interaction; mean and gap of the two internal
  scores; repeat-failure ratio and failures per organisation; secondary-failure flags;
  `LongTermOrLongRecovery`; and an `AggravatingFactorCount` over the rare high-lift flags.
* No date features: severity is flat over time and they would only add drift risk.

### 3. Model (`model.py`, `train.py`)

* **Main model:** LightGBM multiclass classifier with balanced class weights, shallow trees and
  subsampling (2,000 rows is small).
* **Baseline:** multinomial logistic regression on the same features, for reference.
* **Evaluation:** 5-fold stratified cross-validation; all reported metrics are out-of-fold.
* **Decision layer:** `P(severe) = P(4)+P(5)+P(6)`; the case is progressed when
  `P(severe) ≥ τ`; the reported score is the most likely class *within* the chosen band, so the
  score and the decision never disagree.
* **Threshold τ** is chosen on out-of-fold predictions to minimise
  `FN_cost × missed severe + FP_cost × unnecessary investigations` subject to a floor of 95% recall on
  severe cases. The default cost ratio is 5:1.

### 4. Results (5-fold out-of-fold, 2,000 training cases)

| | Logistic baseline | LightGBM, argmax rule | **LightGBM, business rule (τ = 0.05)** |
|---|---|---|---|
| Severe-case recall | 0.92 | 0.92 | **0.98** |
| Missed severe cases | 51 | 50 | **14** |
| Unnecessary investigations | 69 | 40 | 115 |
| Progress rate (true base rate 30%) | 31% | 30% | 35% |
| Exact-score accuracy | 0.74 | 0.80 | 0.78 |
| Within one point | 0.994 | 0.998 | 0.996 |
| Macro F1 | 0.72 | 0.79 | 0.78 |
| Quadratic weighted kappa | 0.94 | 0.95 | 0.95 |
| Severe AUROC | 0.99 | 0.99 | 0.99 |

The business rule trades two points of exact accuracy for a 72% reduction in missed severe cases;
the errors that remain are almost all one point off, and all 14 misses are true 4s whose internal
assessments look like a typical 3.

**Ablation.** Without the four internal assessment fields (`EstimatedImpactScore`,
`EstimatedRiskScore`, `ExpectedFinancialRedressGBP`, `PredictedRemedyBand`) accuracy falls to 0.46 and
severe AUROC to 0.91. The model depends heavily on those upstream assessments; see assumptions.

### 5. Explainability (`explain.py`)

SHAP values from the tree model give a global ranking (internal impact/risk scores, then expected
redress, duration and recovery time, vulnerability × impact, lost income, physical and emotional
impact levels; demographic and organisational fields carry little weight) and a per-case
`TopDrivers` string in the prediction file. The notebook includes global, beeswarm and single-case
plots; figures are in `reports/figures/`.

### 6. Business risk

* The threshold curve and a cost-ratio sensitivity table are in the notebook (section 5). With the 95%
  recall floor, any ratio up to 3:1 selects τ ≈ 0.22 (28 misses, 32% progressed); 5:1 or steeper
  selects τ ≤ 0.05 (≤ 14 misses, 35-40% progressed). The choice is a one-line config change
  (`BusinessCosts` in `config.py` or `--fn-cost/--fp-cost` on the CLI).
* Cases with `P(severe)` between 0.02 and τ are 2.5% of the data but hold 6 of the 14 misses: a
  natural human-review band rather than automatic closure.
* A drift check (KS / chi-square per feature) finds no shift between the training and holdback data,
  so the cross-validated figures are a fair expectation for the holdback predictions.

## Assumptions

1. The four internal assessment fields are produced during the initial review and are available when
   a case is triaged automatically. They are populated in the holdback file, which supports this, but
   if they are not available at triage time the ablation model (`make train-ablation`) must be used
   and τ re-tuned.
2. A missed severe case costs about five times an unnecessary investigation. The framework makes the
   ratio explicit rather than hiding it; the business should confirm it.
3. Blanks in the four "structural None" columns mean `None`; out-of-dictionary numeric values are
   recording errors.
4. The holdback cases come from the same process as the training cases (confirmed empirically by the
   drift check).

## Production readiness

* **Single code path.** Cleaning, features and model live in one saved sklearn pipeline
  (`SeverityModel`); the notebook, CLI and Docker image all call the same functions.
* **Reproducibility.** Fixed seeds, single-threaded LightGBM, pinned requirements, `make all`.
* **Model card.** `models/model_card.json` records the git SHA, training-data hash, library versions,
  configuration, data-quality report and cross-validated metrics of the committed model.
* **Input validation and monitoring hooks.** Every scoring run writes a `*.quality.json` alongside the
  predictions (missing columns, out-of-range counts, unknown categories, inferred values). A rise in
  these, or a move in the progress rate, is the first signal that the input process has changed.
* **Tests.** Cleaning rules, feature determinism, the decision layer's consistency guarantees,
  threshold selection and prediction schema are covered by `pytest`.
* **Suggested operating model.** Automatic closure only well below τ; a human-review band just below τ;
  automatic progression above it. Once outcomes are known, track realised recall on severe cases and
  re-tune τ or retrain when it drifts.

## Limitations and next steps

* 2,000 rows limits hyper-parameter search; the parameters are conservative defaults, not tuned.
* Probabilities are not calibrated (balanced class weights inflate severe probabilities); τ is tuned
  on the same out-of-fold probabilities so the decision is unaffected, but a calibrated `P(severe)`
  would be easier to communicate. Isotonic calibration on a held-out fold is the natural next step.
* A formal ordinal objective or conformal prediction intervals would give each case a guaranteed
  score range rather than a point estimate.
* A fairness audit across `AgeBand` and `PrimaryVulnerability` on realised outcomes should precede any
  fully automatic closure.

## Repository layout

```
data/                     original workbook + CSV exports
src/severity_triage/      config, data, features, model, evaluate, explain, train, predict
tests/                    pytest suite
notebooks/                severity_analysis.ipynb (executed)
models/                   trained model, metrics, model card, ablation model
predictions/              holdback_predictions.csv (+ data-quality report)
reports/figures/          figures produced by the notebook
```
