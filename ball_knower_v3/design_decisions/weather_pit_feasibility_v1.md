# Ball Knower v3 — PIT Weather Data Feasibility v1

Date: 2026-09-14

Status: feasibility resolved; weather remains TEST.

## Question

Does a historically reproducible point-in-time weather-forecast dataset exist that is good enough to permit Ball Knower to test weather without substituting realized postgame conditions?

## Research conclusion

Yes, **for a substantial modern NFL window**.

NOAA archives operational numerical weather prediction forecasts with explicit model initialization cycles and forecast horizons. Most importantly for U.S. NFL venues:

- NOAA HRRR is an operational, hourly updated, 3-km forecast model;
- the AWS NOAA HRRR archive states that historical HRRR data are available **since 2014**;
- NOAA documentation lists HRRR operational versions beginning in 2014 and archives of forecast output;
- RAP provides an older operational forecast family beginning in 2012, although model/version harmonization would be required if used to extend the window.

These are archived **forecast model runs**, not retrospective realized weather observations. Their initialization time and forecast lead therefore provide a supportable approximation of what an operational numerical model predicted before kickoff.

## Final classification

### LOCK

- Weather used as a pregame predictive feature must come from a forecast/model cycle issued before the Ball Knower forecast timestamp.
- Realized game weather, reanalysis and postgame station observations may be used for diagnostics/outcome description but may not masquerade as the pregame forecast.
- Preserve model family/version, initialization time, valid time, forecast lead, grid/location extraction method and acquisition provenance.
- A later archive download may be used for historical replay only under the archive-semantics rules: the archived record must itself identify a forecast generated before the historical cutoff.

### BASELINE

No weather feature is promoted into the first game-total baseline.

The canonical initial margin/total model remains weather-free.

### TEST — feasible weather challenger

Build the first PIT-safe weather experiment over the **HRRR-supported modern window (2014+)** rather than forcing a heterogeneous reconstruction back to 2010.

For each game/forecast origin:

1. identify venue coordinates and outdoor/closed-roof applicability using PIT-safe venue information;
2. select the latest eligible HRRR initialization available **before the Ball Knower forecast timestamp**;
3. extract forecast values valid near scheduled kickoff;
4. preserve forecast lead and model version;
5. derive a small pre-registered candidate set such as wind, precipitation and temperature;
6. evaluate incrementally against the weather-free structural total model chronologically.

Roof/open-roof status must not be retrospectively inferred from the eventual game state when that information was not known at the forecast timestamp.

## Why 2014+ is preferable for the first test

Ball Knower's modeling history reaches earlier than HRRR. It would be possible to investigate RAP/NAM or other older forecast archives, but forcing multiple model systems into one weather series creates avoidable version, resolution and calibration heterogeneity before weather has demonstrated any incremental value.

The simplest defensible experiment is therefore:

- keep the main baseline history unchanged;
- evaluate weather as a challenger on the common modern HRRR window;
- expand earlier only if weather shows robust value.

## Forecast-cycle selection

Do not use a single universal 'game weather' row.

Weather eligibility is keyed to the Ball Knower forecast origin. A Tuesday forecast and a Sunday forecast for the same game should be allowed to see different meteorological forecast cycles if the system is later evaluated at both decision times.

For the normal Tuesday/Wednesday betting workflow, HRRR's short horizon may not extend far enough to every Sunday/Monday game. Therefore HRRR is especially suitable for **near-kickoff or later-week decision points**, not automatically for early-week forecasts.

For longer leads, a global/medium-range operational archive such as GFS can be investigated as a separate TEST source. Do not silently splice GFS and HRRR as though they are one calibrated forecast product.

## Important limitation

This resolves **data feasibility**, not predictive value.

Historical evidence shows weather can affect scoring, but Ball Knower still needs to demonstrate incremental chronological improvement after its structural team-state model. The existence of an archive is not evidence for promotion.

## Promotion requirements

Weather may move from TEST only if:

- historical forecast provenance passes PIT audit;
- feature extraction is reproducible and model-version aware;
- improvement appears in chronological total-distribution proper scores/calibration, not only regression coefficient significance;
- gains survive multiple seasons and are not driven by a few extreme games;
- the comparison uses the same forecast decision time for weather and all other inputs.

## Evidence classes

- **D:** NOAA/NCEI and NOAA Open Data documentation for operational HRRR/RAP forecast archives and model cycles.
- **A/C:** historical NFL weather/scoring evidence supports plausibility but not current incremental value.
- **E:** choose HRRR 2014+ as the first homogeneous PIT-safe Ball Knower experiment window.

## Key sources

- NOAA High-Resolution Rapid Refresh (HRRR) Model, Registry of Open Data on AWS — operational HRRR description and archive since 2014.
- NOAA/NCEP HRRR documentation — operational implementation history.
- NOAA/NCEI Rapid Refresh/Rapid Update Cycle documentation — RAP forecast generation and archive access.
- Borghesi (2008), *Weather biases in the NFL totals market* — historical NFL weather/scoring evidence, insufficient by itself for baseline promotion.