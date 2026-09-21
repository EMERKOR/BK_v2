# Ball Knower v3 — Challenger Research Roadmap

Date: 2026-09-20
Status: development roadmap; retrospective research only unless separately versioned and prospectively registered

## Purpose

This roadmap opens a separate challenger-research track after the frozen-baseline training diagnostic pass. It does not revise `phase3c_prospective_experiment_contract_v1`, `phase3c_prospective_publication_protocol_v2`, or the frozen Phase 3B prospective candidate space.

The frozen baseline continues unchanged on its own prospective stream. Challenger results are development evidence unless and until an explicit later design revision creates a new versioned prospective contract.

Canonical design authority remains:

1. `DESIGN_LOCKS.md`
2. `DESIGN_ADVERSARIAL_REVIEW_2026-09-14.md`
3. `DESIGN_DECISION_RECONCILIATION.md`
4. supporting `design_decisions/` files

Use only `LOCK`, `BASELINE`, `TEST`, and `DEFER` for model-status language.

## Separation of workstreams

### Workstream A — frozen prospective baseline

No challenger work may modify the frozen contract, candidate families, features, priors, penalties, fitting cadence, source eligibility, draw counts, seeds, PMF construction, metrics, promotion gates, registry rules, or publication protocol.

At each admitted origin, Workstream A resolves exact eligible source identities, refits the frozen families causally, builds the forecast bundle, attests it, registers it if all publication conditions pass, and scores it only after outcomes become available.

Prospective outcomes from Workstream A are append-only evidence. Once those outcomes influence a challenger revision, they become development evidence for that revised challenger and cannot be treated as untouched promotion evidence for it.

### Workstream B — challenger research

Workstream B may use retrospective, PIT-safe, chronological development evidence to investigate measured weaknesses and predeclared TEST ideas. It must not write into the prospective registry or represent retrospective results as prospective evidence.

Each challenger experiment should have a small versioned specification written before its evaluation output is examined whenever practical. At minimum freeze:

- research question;
- baseline comparator;
- eligible data window and PIT rules;
- candidate variants;
- tuning procedure;
- chronological split/origin policy;
- primary metrics;
- calibration diagnostics;
- failure criteria;
- promotion criteria for advancing to the next research stage.

Do not widen a candidate family after seeing its held-forward results under the same experiment identifier.

## Priority 1 — Phase 3B hyperparameter identification

Status: `TEST`

### Research question

Can a richer, still-causal Phase 3B search identify persistence, process noise, observation scale/tail behavior, initialization, and offseason transitions separately enough to improve downstream forecast quality without introducing instability or overfitting?

### Why this is first

The diagnostic pass selected the same edge candidate at every audited origin. The current two-point registered space therefore does not identify which underlying state-transition or observation assumptions are responsible for the selection.

### Retrospective experiments allowed now

1. **One-factor sensitivity grid**
   - vary offense/defense persistence around the frozen values while holding the remaining configuration fixed;
   - repeat separately for process SD, observation scale, Student-t degrees of freedom, and offseason persistence/innovation terms;
   - use chronological expanding-window origins only.

2. **Factorial coarse grid**
   - construct a deliberately bounded grid over the parameters that show material sensitivity in the one-factor pass;
   - cap dimensionality before evaluation;
   - keep the observation definition and structural architecture unchanged.

3. **Continuous hierarchical fit prototype**
   - only after the finite-grid geometry is understood;
   - compare a principled continuous hyperparameter treatment against the finite search;
   - require explicit identifiability, posterior-correlation, boundary, and parameter-recovery diagnostics.

### Required diagnostics

- candidate-selection stability by origin;
- objective surface / profile shape;
- parameter recovery on simulation where truth is known;
- posterior or profile dependence among persistence, process noise and observation scale;
- state ranking/spread stability;
- posterior uncertainty calibration;
- downstream margin/total CRPS and coverage using only held-forward chronological games.

### Non-goals

Do not add QB, weather, pace, rest, travel, injury or market features in this stage. The goal is to determine whether the core latent-state dynamics are adequately identified.

## Priority 2 — margin distribution shape and key numbers

Status: `TEST`

### Research question

Can an explicitly discrete or empirically shaped margin distribution recover realistic exact mass at 3 and 7 while improving proper distributional scores rather than merely forcing football-looking key-number probabilities?

### Candidate ladder

Predeclare a small ladder before evaluation:

1. frozen smooth Student-t location model as comparator;
2. empirical residual PMF / shrinkage residual distribution conditional on structural mean;
3. low-complexity discrete residual model with partial pooling;
4. coherent exact-score or bivariate score model only if simpler discrete challengers fail.

A direct post-hoc 3/7 multiplier may be included only as a diagnostic TEST comparator, not as the default challenger.

### Primary evidence

- margin CRPS;
- realized-integer log score;
- exact predicted versus observed mass at 3 and 7;
- cover/push/lose calibration at whole-number thresholds in a separately versioned market-evaluation layer if authorized;
- calibration outside 3/7 to detect mass stealing;
- slice stability by season phase.

### Failure conditions

Reject a shape challenger if key-number calibration improves only by degrading overall CRPS/log score materially, if mass is mechanically moved from nearby outcomes without predictive justification, or if the effect disappears chronologically.

## Priority 3 — margin undercoverage and heteroskedasticity

Status: `TEST`

### Research question

What causes the approximately 0.81 observed 90% margin coverage in the retrospective development sample, and can a predeclared variance/distribution challenger correct it without sacrificing sharpness or introducing post-hoc calibration leakage?

### Candidate ladder

1. residual-scale sensitivity within the existing family;
2. simple heteroskedastic scale regression using only structural pregame quantities already permitted in the challenger specification;
3. Student-t tail/scale alternatives;
4. prior-time-only distributional recalibration as a separate TEST, not folded invisibly into the model.

### Required diagnostics

- 50/80/90% coverage;
- interval width/sharpness;
- PIT deciles and PIT moments;
- CRPS and integer log score;
- calibration by predicted-variance bucket;
- calibration by season phase;
- robustness to early small training prefixes.

Do not tune directly to hit nominal coverage on the same held-forward sample used for evaluation.

## Priority 4 — environment/intercept parameterization

Status: `TEST`

### Research question

Can the margin HFA and total scoring-environment effects be parameterized so they are better identified relative to intercepts while preserving the intended predictive quantity and strict causality?

### Candidate experiments

- centered environment predictors with explicit reference levels;
- reparameterized intercept plus deviation form;
- stronger but defensible weakly informative priors tested through prior-predictive sensitivity;
- dynamic environment simplifications if the historical range cannot identify separate terms.

### Evaluation

Judge the predictive combination first, then parameter geometry. Require coefficient stability, posterior correlation diagnostics, Hessian/Laplace geometry, chronological CRPS/log score, and calibration. Do not prefer a parameterization merely because coefficients look more interpretable.

## Priority 5 — deferred football-context TEST features

These remain `TEST` and should begin only after the first four structural/distributional questions have bounded results.

Suggested order:

1. QB representation ladder already defined by the canonical QB promotion protocol;
2. PIT-safe weather forecasts, primarily for total;
3. rest / short-week effects;
4. travel, time zone and international travel;
5. pace / expected possessions;
6. PROE / pass-rate tendency;
7. injuries / availability beyond the QB-specific work.

Each feature enters as an incremental challenger against the strongest then-current development comparator. Do not bundle several plausible football features into one first test because failure would be uninterpretable.

## Retrospective evidence that may be used

Allowed:

- exact historical source versions satisfying the existing PIT/provenance rules;
- chronological expanding-window replay;
- simulation and parameter-recovery experiments;
- retrospective development scorecards already labeled non-prospective;
- pre-outcome source archives whose historical availability is trustworthy under the canonical provenance rules.

Not allowed as clean challenger evaluation evidence:

- random production train/test splits;
- future smoothing as a historical forecast input;
- realized weather standing in for forecast weather;
- sportsbook inputs in the structural football state;
- Week 3 prospective outcomes used to tune a challenger and then counted as untouched prospective validation for that challenger;
- silent pooling of evidence from different model-development versions.

## Versioning convention

Use separate research artifacts from the prospective baseline.

Recommended layout:

`ball_knower_v3/challenger_research/<experiment_id>/`

Each experiment should contain, where applicable:

- `EXPERIMENT_SPEC.md` frozen before held-forward evaluation;
- machine-readable config/candidate-space file;
- source/provenance manifest;
- deterministic execution metadata;
- evaluation artifacts;
- `RESULTS.md` written after the frozen experiment is run.

The experiment identifier must change when candidate definitions, tuning rules, data windows, metrics or evaluation rules change after results are known.

No challenger artifact should be written to `prospective_registry.json` or use `registered_prospective` evidence status unless a later explicit design revision creates and freezes a new prospective contract for that challenger.

## Suggested first implementation sequence

1. Freeze `phase3b_hyperparameter_identification_v1` before computing its evaluation table.
2. Build the one-factor sensitivity runner using the existing causal replay interfaces rather than a parallel architecture.
3. Add simulation-based recovery tests for state persistence/process/observation combinations.
4. Run the bounded retrospective experiment and write results as development evidence.
5. Use those results to decide whether a coarse factorial search is justified.
6. Only then specify the first margin-shape experiment.

## Promotion discipline

A challenger can advance through research stages because of retrospective evidence, but retrospective success does not make it prospective evidence and does not replace the frozen Phase 3C promotion gate.

Any eventual production promotion requires an explicit design reconciliation, a new frozen model-development/prospective contract where required, and a future untouched prospective stream. The current frozen baseline remains the active baseline unless such a revision is authorized.