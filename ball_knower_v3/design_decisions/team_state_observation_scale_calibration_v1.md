# Ball Knower v3 — Team-State Observation Scale Calibration v1

Date: 2026-09-15

Status: resolved implementation clarification for Design Lock 7.

## Question

Should the current team-state `observation_sd = 1.0` smoke-test default be replaced by the approximately `1.38` empirical residual standard deviation observed in preliminary inspection?

## Resolution

No. Neither `1.0` nor `1.38` is promoted as a football constant.

The baseline robust team-state model uses a Student-t observation likelihood. Its scale parameter is not the same quantity as the marginal residual standard deviation. For Student-t degrees of freedom `nu > 2`,

`SD = scale * sqrt(nu / (nu - 2))`.

Therefore a raw comparison between a Student-t scale parameter and an empirical residual standard deviation is invalid unless `nu` and the conditioning model are accounted for.

The approximately `1.38` pooled residual standard deviation is also not a direct estimate of play-level observation noise unless it is measured conditional on the fitted latent offense, defense, league intercept, and applicable state uncertainty. A crude residual distribution can contain genuine between-team/state variation that belongs in the latent state rather than the observation-noise term.

## LOCK

- Observation uncertainty and latent process/state uncertainty remain distinct and must be separately identified.
- Do not replace the observation scale with a pooled/raw EPA residual standard deviation.
- Do not interpret `StateSpaceConfig.observation_sd` as the marginal standard deviation of raw EPA when the robust model uses Student-t observation treatment.
- Hyperparameter fitting at a historical forecast origin may use only prior-time data.
- Uncertainty calibration must be validated with causal one-step-ahead / forecast-origin predictive residuals and posterior predictive coverage, not by matching one raw residual-SD number.

## BASELINE

- Learn/tune the play-level Student-t observation scale jointly with the other team-state hyperparameters from prior-time training evidence.
- Estimate Student-t tail thickness under the existing regularized policy when computationally stable; otherwise use a pre-registered training-only fixed value with sensitivity analysis.
- Evaluate standardized predictive innovations, posterior predictive EPA distribution/tails, interval coverage, and synthetic parameter recovery for process/observation separation.

## Engineering consequence

The existing `StateSpaceConfig.observation_sd = 1.0` value remains a smoke-test/interface default only. It is not a scored-production setting.

The next hyperparameter-fitting implementation must not contain logic such as:

`observation_sd = raw_epa_residuals.std()`

unless those residuals arise from the appropriate prior-time conditional predictive model and the Student-t scale conversion/likelihood is handled correctly. Preferred implementation is direct fitting/scoring of the model's observation-scale parameter rather than a plug-in raw-SD substitution.

## Prior audit correction

Retract any claim that `1.0` versus approximately `1.38` by itself proves uncertainty is understated by about 26 percent. That comparison mixes parameterizations and may mix latent team-strength variation with observation noise.

The correct conclusion is:

> Observation-scale calibration remains an implementation-validation requirement. The pooled residual SD cannot be directly substituted for a Student-t scale parameter; observation scale, tail thickness, and latent process uncertainty must be learned/validated together under the chronological replay contract.

## Relationship to existing canonical decisions

This clarification does not change a model status. It operationalizes existing locks in `DESIGN_LOCKS.md` and `team_state_implementation_contract_v1.md` requiring:

- separate process and observation uncertainty;
- prior-time-only hyperparameter learning;
- Student-t robust observation treatment;
- synthetic parameter-recovery tests;
- posterior predictive and uncertainty/coverage diagnostics.

The team-state architecture remains implementation-ready.