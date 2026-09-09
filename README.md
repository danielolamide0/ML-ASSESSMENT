# Complaint severity triage model

This is my submission for the AI/ML engineer technical assessment. I built a classification model that predicts the **Severity Score (1-6)** of a complaint about a public
service and, from it, the triage decision the score drives:

| Severity Score | Outcome |
|---|---|
| 1-3 | Not progressed |
| 4-6 | Progressed for further investigation |

The organisation has a low tolerance for severe cases being classified as low severity, so I built the
solution around that asymmetry rather than around raw accuracy.

## Deliverables

| Item | Where |
|---|---|
| Source code | `src/severity_triage/` (installable package with `severity-train` / `severity-predict` CLIs) |
| Notebook | `notebooks/severity_analysis.ipynb` (executed, with outputs and figures) |
| Holdback predictions | `predictions/holdback_predictions.csv` |
| README | this file |
| Requirements | `requirements.txt` (pinned to the versions used to produce the committed outputs) |
| Tests | `tests/` (26 tests, `make test`) |
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
| `TopDrivers` | the three features that pushed this case most towards or away from "severe" (SHAP), with their observed values; an imputed value is shown as `missing` |

## Quick start

Requires Python 3.11 or later.

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

## My approach

### 1. Data preparation (`data.py`)

I treat the data dictionary in the assessment workbook as the contract and is encoded in
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
* Nominal categoricals are one-hot encoded with the category list fixed from the data dictionary plus
  an explicit `Missing` level, and one missing-value indicator exists for every numeric column, so the
  feature set is identical in every cross-validation fold and at inference; an out-of-dictionary value
  maps to an all-zero row rather than shifting columns.
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

| | Logistic, argmax | Logistic, business rule (τ = 0.26) | LightGBM, argmax | **LightGBM, business rule (τ = 0.03)** |
|---|---|---|---|---|
| Severe-case recall | 0.92 | 0.98 | 0.92 | **0.98** |
| Missed severe cases | 51 | 11 | 46 | **11** |
| Unnecessary investigations | 69 | 130 | 40 | 130 |
| Progress rate (true base rate 30%) | 31% | 36% | 30% | 36% |
| Expected cost per case (FP units) | 0.16 | 0.09 | 0.14 | 0.09 |
| Exact-score accuracy | 0.74 | 0.73 | 0.80 | 0.77 |
| Within one point | 0.994 | 0.993 | 0.999 | 0.997 |
| Macro F1 | 0.72 | 0.71 | 0.79 | 0.77 |
| Quadratic weighted kappa | 0.94 | 0.94 | 0.95 | 0.95 |
| Severe AUROC | 0.99 | 0.99 | 0.99 | 0.99 |

Two honest readings of this table:

* **The decision rule matters more than the model.** Under the same cost-minimising rule the logistic
  baseline and LightGBM reach the same triage outcome (11 missed, 130 unnecessary investigations).
  For the binary progress decision alone the simpler model would do.
* **LightGBM earns its place on the score itself**, which the prediction file has to report: six
  points more exact-score accuracy and a lower MAE, plus per-class SHAP explanations.

The business rule trades 2.5 points of LightGBM's exact accuracy for a 76% reduction in missed severe
cases (46 → 11); the errors that remain are almost all one point off, and all 11 misses are true 4s
whose internal assessments look like a typical 3.

τ is chosen on out-of-fold predictions and reported on the same predictions, which is mildly
optimistic: a nested check (choose τ on four fifths, evaluate on the fifth, 20 splits) gives mean
severe recall 0.97 against 0.98 reported, and the chosen τ ranges 0.03-0.13 across splits. The exact
value is not sharply identified by 2,000 cases; that it belongs far below 0.5 is.

**Ablation.** I also trained the model without the four internal assessment fields
(`EstimatedImpactScore`, `EstimatedRiskScore`, `ExpectedFinancialRedressGBP`, `PredictedRemedyBand`).
Accuracy falls to 0.45 and severe AUROC to 0.91. The model depends heavily on those upstream
assessments; see the assumptions section.

### 5. Explainability (`explain.py`)

SHAP values from the tree model give a global ranking (internal impact/risk scores, then expected
redress, duration and recovery time, vulnerability × impact, lost income, physical and emotional
impact levels; demographic and organisational fields carry little weight) and a per-case
`TopDrivers` string in the prediction file. The notebook includes global, beeswarm and single-case
plots; figures are in `reports/figures/`.

### 6. Business risk

* The threshold curve and a cost-ratio sensitivity table are in the notebook (section 5). With the 95%
  recall floor, ratios of 1:1 and 2:1 select τ = 0.21 (29 misses, 32% progressed), 3:1 selects 0.13
  (25 misses, 33%), and 5:1 or steeper selects τ ≤ 0.03 (≤ 11 misses, 36-40% progressed). The choice
  is a one-line config change (`BusinessCosts` in `config.py` or `--fn-cost/--fp-cost` on the CLI).
* Out of fold, cases with `P(severe)` between 0.01 and τ are 3.6% of the data but hold 6 of the 11
  misses: a natural human-review band rather than automatic closure.
* A drift check (KS / chi-square per feature) finds no shift between the training and holdback data,
  so the cross-validated figures are a fair expectation for the holdback predictions.

## Assumptions I made

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

* **Single code path.** Cleaning is one deterministic, dictionary-driven function; features and model
  live in one saved sklearn pipeline (`SeverityModel`). The notebook, CLI and Docker image all call
  the same two.
* **Reproducibility.** Fixed seeds, single-threaded LightGBM, pinned requirements, `make all`.
* **Model card.** `models/model_card.json` records the git SHA of the code the model was trained
  with (suffixed `-dirty` if the tree had uncommitted changes), the training-data hash, library
  versions, configuration, data-quality report and cross-validated metrics.
* **Input validation and monitoring hooks.** Every scoring run writes a `*.quality.json` alongside the
  predictions (missing columns, out-of-range counts, unknown categories, inferred values). Duplicate
  or missing case references are rejected. A rise in these counts, or a move in the progress rate, is
  the first signal that the input process has changed.
* **Fail safe.** A case whose probabilities cannot be computed is progressed, never closed
  automatically.
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
* The threshold is selected and reported on the same out-of-fold predictions (see results); with more
  data, select it on a separate validation split.
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
