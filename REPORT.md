# Symptom Recommender: Experimental Report

This study compares recommendations from historical symptom patterns with a hybrid that adds clinical knowledge, then tests semantic models for user-text normalization. Outcomes measure recovery of recorded symptoms, not diagnostic accuracy.

<a id="dedup-diagnostics"></a>
<a id="hitrate-diagnostics"></a>

## Data and evaluation design

The dataset contains 1,000 sessions and 983 unique rows, with sex, age, typed text, confirmed/denied symptoms, and HPI answers. The first confirmed symptom becomes the CC. A chief-complaint-stratified split with seed 42 assigns 807 sessions to training and 193 to test, keeping exact-copy groups together. Aliases and α are learned using training data only.

Only **83 test sessions have associated symptoms**, producing **96 hidden-symptom cases**: 73 sessions contribute one, eight contribute two, one contributes three, and one contributes four. All models use the same fixed cases and reference mappings:

- **Hide-one:** hide one associated symptom while showing the others and their own HPI.
- **First-step:** show only the CC and its HPI, then evaluate each associated symptom as a target.
- **HitRate@K:** the fraction of hidden symptoms found in the top K. The primary endpoint is @5; intervals use 1,000 paired bootstrap draws of whole sessions.

Shown symptom names still pass through the normalizer, but ranking tests start from **selected symptoms in the record**, not raw user queries. Typed-text normalization is evaluated separately. Removing all exact-copy rows and applying the same splitting procedure left HitRate@5 unchanged in the deduplication experiment.

<!-- report:evaluation:start -->
<a id="evaluation"></a>

## Default recommender and normalizer evaluation

| Method | Hide-one @3 | Hide-one @5 | Hide-one @8 | First-step @5 |
|---|---:|---:|---:|---:|
| Popularity | 33.33% | 34/96 (35.42%) | 53.12% | 34/96 (35.42%) |
| Co-occurrence | 75.00% | 77/96 (80.21%) | 83.33% | 76/96 (79.17%) |
| Engine data (α = 0) | 78.12% | 80/96 (83.33%) | 86.46% | 79/96 (82.29%) |
| Book only (α = 1) | 29.17% | 41/96 (42.71%) | 57.29% | 43/96 (44.79%) |
| Hybrid (α = 0.1) | 78.12% | 81/96 (84.38%) | 86.46% | 80/96 (83.33%) |

The hybrid differs from co-occurrence by +4.17 percentage points on hide-one. The paired bootstrap 95% CI for that difference includes zero (hide-one: -2 to +11 percentage points; first-step: -1 to +9 percentage points). Compared with the same engine at α = 0, the net difference is +1 case(s). The raw-label baseline comparison does not isolate the contribution of the KB.

Giving each session equal weight yields hide-one @5 of 86.14% versus 82.03% for the baseline. This is reported separately from the primary metric, which weights each hidden case equally.

### Text normalization

Compare symptoms extracted from typed text with confirmed symptoms in the same session:

| Test text group | Matches a recorded symptom |
|---|---:|
| Same as a recorded name after cleaning | 162/176 |
| Different from recorded names; seen in training | 64/96 |
| Different from recorded names; unseen in training | 10/19 |

These are session-consistency proxies, not physician-labelled accuracy: records may be incomplete, and this metric does not penalize all extra predictions. At least one HPI answer was parsed in 150/465 items. That measures coverage of supported questions, not parsing correctness.

### Clinical regression

203 vignettes: top-three expectations 146/147; safety, flow and other expectations 96/96; book diagnoses in the top three 15/19

Unmet top-three expectation: cough_ace_inhibitor. The leiomyoma book case ranks fourth; examination and ultrasound are outside the current input. Vignettes are regression checks and have informed development, so they are not independent clinical validation.
<!-- report:evaluation:end -->

<a id="retrieval-development"></a>

## Semantic normalization experiments

Semantic fallback runs only when existing rules cannot assign a symptom. Both methods preserve exact matches, ambiguity, and present/absent/unsure/historical states. Inference is local, and settings are selected within training folds before comparison with the original test.

<!-- report:semantic-validation:start -->
<a id="semantic-validation"></a>

## E5-small training calibration

Use `intfloat/multilingual-e5-small` on CPU as a fallback after rules. Select thresholds using five grouped training folds, relearning aliases within each fold.

There were 73 fallback queries; the reference appeared in the top three candidates in 43/73 queries. Selection requires at least five accepted queries and ≥90% proxy agreement.

The selected policy accepted 0 queries with 0 agreements. It therefore supplies candidates only. Calibration results select the policy; they are not an independent accuracy estimate.
<!-- report:semantic-validation:end -->

<!-- report:semantic-evaluation:start -->
<a id="semantic-evaluation"></a>

## E5-small evaluation

| Proxy | Rules | E5-small |
|---|---:|---:|
| seen: matches record | 64/96 | 64/96 |
| unseen: matches record | 10/19 | 10/19 |
| Reference in top-three candidates when deferred | 10/32 | 12/32 |
| hide_one HitRate@5 | 81/96 | 81/96 |
| first_step HitRate@5 | 80/96 | 80/96 |

Candidate agreement improves, but automatic mapping and HitRate do not. Candidate consistency with records is measured separately from automatic symptom assignments.
<!-- report:semantic-evaluation:end -->

<!-- report:retrieval-evaluation:start -->
<a id="retrieval-evaluation"></a>

## E5-base retrieval evaluation

Fit a ridge classifier to E5-base embeddings of KB names and aliases. Select its gate using five folds of 677 training sessions, excluding all sessions containing the 49 review-validation phrase groups. Selection yields 16/16 proxy agreements from 6 distinct phrase groups. No policy meets the acceptance criteria on novel development queries alone.

| Proxy | Rules | E5-base + ridge |
|---|---:|---:|
| seen: matches record | 64/96 | 68/96 |
| unseen: matches record | 10/19 | 10/19 |
| hide_one HitRate@5 | 81/96 | 81/96 |
| first_step HitRate@5 | 80/96 | 80/96 |

There are 8 additional positive assignments: 5 match a recorded symptom and 3 cannot be verified from the record. On unseen review-holdout text, agreement is 35/77 versus 35/77.

Gains on previously seen text do not establish accuracy on new text. This remains an opt-in experiment; rules remain the default. Details are in `build/retrieval/evaluation.json`.
<!-- report:retrieval-evaluation:end -->

<a id="normalization-assertions"></a>
<a id="closeout"></a>

## System verification and conclusions

All 272 tests, lint, and KB validation pass. Docker book-only, hybrid/rules, and hybrid/retrieval each matched the local API on 7/7 invented requests and passed health checks. Hybrid/retrieval also worked with networking disabled and read-only files.

**The default is hybrid α = 0.1, selected on training data, with rules normalization.** It scores highest among the main ranking methods tested. Semantic experiments do not improve agreement on unseen text or recommendation HitRate. The hybrid's difference interval against the baseline includes zero, so superiority is not established.

Records are influenced by the original symptom picker, with no final diagnoses or physician-labelled normalization gold. The 197 development and 49 validation phrase groups remain unreviewed. All 25 KB chapters are drafts, and vignettes have informed development; they do not establish independent clinical validity. The original test has also been inspected during analysis, so these comparisons remain exploratory.

Confirmation requires physician-labelled text and fresh independent data after freezing the method; no new dataset is currently available. Reproduction commands and annotation instructions are in [README.md](README.md).
