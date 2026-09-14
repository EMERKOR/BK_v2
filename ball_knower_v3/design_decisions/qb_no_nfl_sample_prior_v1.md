# Ball Knower v3 — No-NFL-Sample Quarterback Prior v1

Date: 2026-09-14

Status: resolved design decision.

## Question

How should Ball Knower initialize quarterbacks with essentially no meaningful NFL play-by-play sample — especially rookies — without letting a generic replacement prior dominate too strongly or introducing leakage from draft/college information?

## Decision summary

For quarterbacks with no meaningful NFL action-play sample, Ball Knower should use a **hierarchical pre-NFL prior centered primarily on draft position**, with **wide uncertainty**. Draft capital is the baseline public signal because peer-reviewed research finds that earlier-selected quarterbacks perform better on average in the NFL and that the draft aggregates substantial qualitative and quantitative pre-draft information.

College/combine features are not part of the baseline prior because their incremental predictive value is inconsistent across the literature. They remain `TEST`, with college rushing ability the strongest specifically supported candidate for incremental value.

The prior must be constructed from information actually available by the forecast date. Draft position may only be used after the player is drafted. Pre-draft forecasts, if Ball Knower ever produces them, require a separate model.

Conceptually:

`Q_q,0 ~ Normal(mu_draft(draft_slot_q), sigma_draft(draft_slot_q))`

where both the mean and uncertainty are estimated historically and shrink smoothly across draft position rather than using arbitrary round buckets.

Once NFL action-play evidence arrives, the player's posterior updates normally through the dynamic QB model and the influence of the prior fades.

---

# Evidence classification

- **A — peer-reviewed NFL evidence:** draft position has measurable association with subsequent quarterback performance; pre-draft college/combine metrics have inconsistent incremental predictive value.
- **B — statistical theory:** hierarchical priors and partial pooling are appropriate when direct player-level evidence is absent or sparse.
- **C — public football evidence:** rookie/QB evaluation work supports large uncertainty and rapid posterior updating once NFL evidence arrives.
- **E — Ball Knower design inference:** use draft slot as the main prior mean signal while allowing variance to remain large and empirically estimated.

---

# 1. Baseline prior

## BASELINE — draft-position-conditioned hierarchical prior

For a quarterback with no meaningful NFL sample, initialize the QB latent state from a smooth prior conditioned on actual NFL draft position.

Do not use simple round labels if the historical data support a smoother mapping. A player drafted 3rd overall and one drafted 31st overall should not automatically receive the same prior merely because both were first-round selections.

The prior should estimate both:

- expected initial QB effect as a function of draft slot;
- uncertainty around that expectation.

The uncertainty must remain substantial even for very high draft picks. Draft position contains useful information but does not make rookie outcomes precise.

## LOCK — draft capital is informative, not determinative

Peer-reviewed work finds that earlier drafted quarterbacks tend to perform better on average, but quarterback outcomes remain highly variable.

Therefore:

- high draft capital may shift the prior mean upward;
- it must not collapse posterior uncertainty;
- low draft capital may shift the prior downward but must not prohibit strong NFL updating later.

A first-overall pick is not assigned an elite NFL state by rule, and an undrafted player is not prevented from moving rapidly when NFL evidence supports it.

---

# 2. Why draft position is the baseline signal

## Evidence

Wolfson, Addona & Schmicker (2011) studied drafted quarterbacks and concluded that college and combine statistics added little predictive value for NFL success, while NFL teams' draft decisions appeared to aggregate available pre-draft information reasonably effectively.

Boulier et al. found that quarterbacks taken earlier in the draft performed better on average than quarterbacks taken later.

Craig & Winchester (2021) found that among drafted quarterbacks, college rushing ability predicted later NFL performance, while college passing measures were much less consistently useful once selection had occurred.

Taken together, these results support treating **draft slot as the strongest simple public summary prior** and treating richer college inputs as challengers rather than baseline requirements.

---

# 3. Replacement prior is not the rookie baseline

## SUPERSEDED BASELINE

The prior design that initialized all no-evidence quarterbacks near a generic replacement/low-usage QB level is too coarse for rookies.

Peer-reviewed `nflWAR` uses low-involvement quarterbacks to define replacement level for valuation, which remains useful for backup/replacement comparisons. But a newly drafted quarterback and an anonymous low-usage veteran backup are not exchangeable information states.

Therefore:

- replacement-level pooling remains relevant for low-usage NFL veterans/backups with minimal evidence;
- drafted rookies with no NFL sample use the draft-conditioned prior;
- undrafted rookies/no-evidence players use a much broader low-capital/replacement-style prior.

This is a prior distinction, not a permanent player classification.

---

# 4. College and combine information

## TEST — college rushing ability

College rushing ability is the strongest evidence-backed candidate for enriching the rookie prior. Craig & Winchester find it significantly associated with NFL performance among selected quarterbacks even when scout grades are included.

Test candidate features may include:

- opponent-adjusted college rushing efficiency;
- rushing volume/usage;
- sack-adjusted mobility proxies where historically available.

They must demonstrate incremental chronological predictive value beyond draft slot.

## TEST — college passing metrics

College passing efficiency, completion rate, yards per attempt, and related statistics may be tested, but the literature does not justify locking them into v1.

They require competition-strength adjustment and careful era/scheme treatment to avoid false precision.

## TEST — combine / physical measures

Combine athletic measures may be tested, but existing literature does not establish them as strong standalone QB success predictors.

Do not make hand size, forty time, Wonderlic-type measures, or similar combine variables baseline inputs without incremental evidence.

## TEST — public scouting grades

Historically archived scout grades may contain useful information because draft decisions themselves aggregate scouting input. However, consistent historical PIT archives are difficult and provider definitions change.

Use only if the source/version and historical availability can be reconstructed honestly.

---

# 5. Undrafted and low-capital quarterbacks

## BASELINE — broad low-capital prior with high uncertainty

Undrafted/no-draft quarterbacks with no NFL evidence should initialize from a broad low-capital/replacement-style prior, not from league-average starter quality.

The variance should be wide enough that the model can adapt quickly after even a modest NFL sample.

For low-usage veterans with some NFL evidence, their existing NFL posterior takes precedence over draft capital. Draft position should not continue to dominate years into a career.

---

# 6. Prior decay once NFL evidence arrives

## LOCK — NFL evidence supersedes pre-NFL information

The draft-conditioned prior is an initialization mechanism.

Once the quarterback accumulates NFL action plays, the posterior updates through the same dynamic QB model used for veterans. The prior influence should diminish naturally through Bayesian updating rather than through a hand-selected rule such as 'ignore draft capital after Week 6.'

Historical NFL evidence carries forward across seasons according to the QB transition model.

Draft position should not be repeatedly re-added as a feature each week after the player's NFL state already exists unless a separate challenger demonstrates incremental value without double counting.

---

# 7. Point-in-time and leakage rules

## LOCK — draft information becomes available only when it becomes public

For historical reconstruction:

- actual draft slot may be used after the player is selected;
- it may not be backfilled into forecasts made before that draft occurred;
- combine/college/scouting variables must likewise have supportable historical availability dates.

The eventual NFL outcome, eventual starter status, future depth-chart position, later consensus scouting grades, or later re-draft evaluations may never enter the historical prior.

## LOCK — model training itself remains chronological

The mapping from draft slot to initial NFL QB effect must be estimated only using prior draft classes available at the historical forecast cutoff.

A 2015 rookie prior may not be estimated using the careers of quarterbacks drafted in 2018 or 2024.

This rule applies to all richer rookie-prior challengers as well.

---

# 8. Prior functional form

## BASELINE — smooth monotonic relationship, not arbitrary buckets

Start with a regularized smooth relationship between draft slot and expected initial QB effect.

Candidate implementation:

`mu_q = f(log(draft_pick))`

or another empirically selected monotone/smooth transform.

The exact transform is not locked. It must be trained chronologically and compared with simple bucketed alternatives.

The variance may also depend on draft capital, but Ball Knower must not assume top picks are necessarily much less uncertain unless historical evidence supports it.

## TEST

Compare:

1. single replacement prior for all no-sample QBs;
2. round-bucket prior;
3. smooth draft-slot prior — baseline candidate;
4. draft slot + college rushing;
5. draft slot + broader college/combine/scouting features;
6. class/era-specific priors;
7. mixture priors separating drafted starters, low-capital backups and undrafted players.

---

# 9. Evaluation requirements

Evaluate rookie/no-sample priors on genuinely future quarterback classes using only information that would have existed at the time.

Required scorecards include:

- predictive log score / proper score on early-career QB play;
- calibration and interval coverage;
- downstream game-distribution performance when a rookie/unknown QB starts;
- performance in first start / first 50 / first 100 / first 250 action plays;
- robustness by draft range and era;
- comparison with the no-QB or generic replacement prior.

Do not select the prior based on career WAR or hindsight ranking alone if the production task is next-game forecasting.

---

# 10. Research basis

Primary sources used:

- Wolfson, Addona & Schmicker (2011), *The Quarterback Prediction Problem: Forecasting the Performance of College Quarterbacks Selected in the NFL Draft*, Journal of Quantitative Analysis in Sports — draft decisions aggregate meaningful pre-draft information; college/combine metrics add limited predictive value in their analysis.
- Craig & Winchester (2021), *Predicting the national football league potential of college quarterbacks*, European Journal of Operational Research — college rushing ability significantly predicts NFL performance among selected quarterbacks, while college passing measures are weaker after selection.
- Boulier et al. (2010), *Evaluating National Football League draft choices: The passing game*, International Journal of Forecasting — earlier-drafted quarterbacks perform better on average.
- Yurko, Ventura & Horowitz (2019), `nflWAR` — low-involvement quarterback pooling provides a reproducible replacement-level concept, but this is a valuation construct rather than a complete rookie forecasting prior.

No source supports a precise deterministic rookie rating from draft position. The baseline therefore uses draft capital to shift a wide prior rather than to create certainty.

---

# Final classification

## LOCK

- no-sample QB forecasts must remain highly uncertain;
- pre-NFL information may only enter when it was actually available historically;
- historical prior training is chronological by draft class;
- NFL evidence supersedes the initialization prior naturally through posterior updating;
- draft capital is informative but never deterministic.

## BASELINE

- drafted no-NFL-sample QB: smooth draft-position-conditioned hierarchical prior with wide variance;
- undrafted/no-draft no-sample QB: broad low-capital/replacement-style prior;
- low-sample NFL veterans: NFL posterior takes precedence over draft capital.

## TEST

- college rushing metrics;
- broader opponent-adjusted college passing metrics;
- combine measures;
- historically archived scouting grades;
- richer mixture/class/era priors;
- draft-position-dependent prior variance.

## DEFER

- subjective manual rookie grades entered by Ball Knower operators;
- retrospective consensus re-draft grades;
- proprietary scouting/tracking data without historical PIT archives;
- fixed heuristics such as 'first-round QB = +X EPA'.