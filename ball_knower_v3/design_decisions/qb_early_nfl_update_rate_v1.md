# Ball Knower v3 — Early NFL QB Update Rate v1

Date: 2026-09-14

Status: resolved design decision.

## Question

Should rookies and other quarterbacks with very little NFL evidence update faster than established veterans once NFL action begins? In particular, should Ball Knower use a special early-career learning multiplier, temporarily higher process variance, or a different observation-weighting rule?

## Decision summary

Ball Knower should **not use a hand-coded accelerated-learning multiplier for rookie/low-sample quarterbacks in the baseline**.

The baseline uses ordinary Bayesian updating under the same causal QB observation model already selected for veterans, with two important differences that arise naturally from the prior rather than from a special update rule:

1. rookie/no-sample quarterbacks begin with much wider posterior uncertainty;
2. that wider prior allows real NFL evidence to move the posterior materially without artificially overweighting a small number of early plays.

Conceptually:

`posterior_QB_t ∝ likelihood(new NFL plays | QB_state_t) × prior_QB_t`

The early-career prior is wide; the likelihood remains calibrated to the actual noise of NFL quarterback play.

## Why not accelerate the first games?

Public NFL evidence shows that very early rookie performance is noisy and not reliably representative of later rookie or second-year performance.

Practitioner studies find:

- the first few rookie starts can have essentially no correlation with later starts in the same season;
- several rookie QB efficiency and turnover metrics are not stable from Year 1 to Year 2;
- rookie performance is generally less predictive of future performance than later-career seasons;
- draft-position-informed Bayesian forecasts improve as dropback volume accumulates rather than assuming the first games deserve disproportionate weight.

Therefore a rule such as "double the learning rate for the first three starts" would increase sensitivity precisely where observed performance is especially noisy.

## LOCK — evidence volume and uncertainty govern learning

Posterior movement must be driven by:

- prior uncertainty;
- actual number of eligible QB action plays;
- observation noise;
- opponent/context adjustment;
- process uncertainty between state times.

Do not multiply early-career observations by an arbitrary experience-based factor.

A 40-dropback debut is not automatically more informative merely because the quarterback is a rookie.

## BASELINE — common observation model, wider early prior

Rookie and low-sample QB observations use the same robust, opponent-relative, game-state-adjusted observation structure as other QBs.

What differs is the state prior:

- no-NFL-sample rookies begin from the draft-informed prior already locked;
- undrafted/no-evidence QBs begin from the broader low-capital/replacement-style prior;
- posterior uncertainty is materially larger than for established starters.

Because Bayesian updating naturally weights a wide prior less heavily than a precise veteran posterior, meaningful NFL evidence can move a rookie state quickly **without changing the likelihood itself**.

## BASELINE — no special first-N-start cutoff

Do not define a fixed first-3, first-5, first-8, or first-season regime in the baseline.

NFL playing opportunities vary too much for starts to be a clean information unit. One quarterback may accumulate 250 action plays in five starts while another has far fewer meaningful observations.

Use eligible play volume and posterior uncertainty rather than arbitrary start counts.

## TEST — experience-dependent process variance

Young quarterbacks plausibly undergo more real skill change than established veterans because of adaptation, coaching and development.

Therefore Ball Knower should test whether QB process variance depends on experience, for example:

`Var(process_noise_QB) = h(career_action_plays, seasons_experience)`

or through a small number of learned experience bands.

This is a **state-evolution** question, not a reason to overweight observed EPA.

Promotion requires chronological improvement in:

- future QB-state calibration;
- future game-distribution proper scores;
- starter-change forecasts;
- uncertainty coverage.

## TEST — temporary early-career variance regime

A finite high-volatility early-career regime may be tested if historical evidence supports it.

Examples:

- larger process variance during the first NFL season;
- process variance that decays smoothly with career action plays;
- change-point models where uncertainty contracts after sufficient NFL evidence.

No threshold is locked in advance.

## TEST — age/experience-aware transition model

A richer model may allow persistence and process variance to depend jointly on age and NFL experience. This remains TEST because the direct NFL evidence is not strong enough to justify a specific functional form in the baseline.

## LOCK — do not confuse learning uncertainty with true skill volatility

Two distinct uncertainties must remain separate:

1. **epistemic uncertainty** — Ball Knower does not yet know the QB's level because evidence is sparse;
2. **process volatility** — the QB's actual underlying ability may be changing.

A rookie naturally has more epistemic uncertainty. That fact alone does not prove greater process variance.

The model must not encode the same uncertainty twice by using both an extremely wide prior and an unsupported large early-career process multiplier.

## TEST — preseason information

Preseason QB performance remains TEST only. Public analysis finds weak predictive value for rookie preseason performance and notes that preseason offensive conditions differ materially from regular-season football.

Any preseason feature must demonstrate incremental chronological value beyond draft prior and regular-season evidence, with strict point-in-time provenance.

## Research basis

Primary evidence used in this resolution:

- PFF, *What remains stable between a rookie QB's first and second NFL seasons?* — several efficiency and turnover measures were not stable between rookie and second seasons; supports caution against overreacting to early efficiency results.
- PFF, *Preseason NFL football often tells us nothing about rookie quarterbacks* — in the studied rookie sample, first-three-start performance showed essentially no correlation with the remainder of rookie starts; preseason evidence was also weak.
- PFF, Bayesian rookie-QB forecasting work — demonstrates gradual updating from draft-position priors as NFL dropback volume accumulates rather than an explicit early-game multiplier.
- Evans et al. (2025), *NFL Quarterback Development—The Role of Competing Quarterbacks, Rookie Playing Time, and Team Quality* — provides direct evidence that young-QB development is heterogeneous and context dependent, supporting TEST status for experience-dependent process dynamics rather than a universal acceleration rule.
- General Bayesian state-space methodology — wide priors already permit faster posterior movement when evidence is sparse; observation weighting and latent process variance should remain conceptually distinct.

## Final classification

### LOCK

- no arbitrary rookie fast-update multiplier;
- posterior movement is governed by prior uncertainty, play volume, observation noise and process uncertainty;
- epistemic uncertainty and true process volatility remain distinct;
- no fixed first-N-start transition rule.

### BASELINE

- same robust NFL QB observation likelihood for rookies and veterans;
- wider draft-informed/low-capital prior for no-sample QBs;
- ordinary sequential Bayesian updating.

### TEST

- experience-dependent process variance;
- temporary early-career high-volatility regime;
- age/experience-aware persistence;
- preseason information.

### DEFER

- hand-written rules such as "first 3 starts count double" or "rookies improve X EPA per week."
