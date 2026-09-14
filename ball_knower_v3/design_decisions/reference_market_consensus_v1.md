# Ball Knower v3 — Reference-Market Consensus v1

Date: 2026-09-14

Status: implementation baseline resolved; production superiority remains TEST.

## Question

What should Ball Knower use as the first defensible reference-market consensus benchmark, distinct from an executable sportsbook quote?

## Research conclusion

Betting odds are highly informative forecasts, individual bookmakers can differ in forecasting quality, and aggregation across sources can reduce idiosyncratic book-specific error. Research also shows that vig-removal method matters and that Shin-style adjustments can outperform simple normalization in some sports/markets.

None of the evidence establishes one universally optimal NFL multi-book consensus recipe.

Therefore the correct resolution is:

- define a conservative, auditable **BASELINE benchmark recipe**;
- keep alternative vig-removal, weighting and interpolation methods as `TEST`;
- do not promote the exact recipe as a production truth until Ball Knower compares alternatives chronologically.

## LOCK — reference consensus is not an executable offer

The reference market exists to answer:

> what did the broader market imply at this forecast time?

It does not answer:

> what price could Ball Knower actually bet?

Executable wager evaluation always uses the actual sportsbook line/price and timestamp.

## LOCK — timestamp alignment

All quotes entering a consensus snapshot must be supportably available at or before the same Ball Knower as-of time.

Do not mix an early quote from one book with a later quote from another and call the result a contemporaneous consensus unless the snapshot policy explicitly allows/records staleness.

Preserve book update/source time where available.

## LOCK — never aggregate different betting thresholds as though they are the same probability

A probability of home cover at `-2.5` is not directly comparable to home cover at `-3`.

Therefore Ball Knower may not average cover probabilities across different spread lines or over probabilities across different total thresholds without an explicit interpolation/distribution model.

Interpolation across lines remains `TEST`.

## BASELINE — same-threshold, equal-weight robust consensus

For a requested spread/total threshold `X`:

1. collect eligible books quoting **that same threshold X** at the as-of time;
2. pair the two sides of the market from the same book/snapshot;
3. convert American/decimal prices to raw implied probabilities;
4. remove overround within each book using simple multiplicative normalization for the first benchmark;
5. aggregate the resulting fair probabilities across books with an **equal-weight median**;
6. report the number of contributing books and dispersion/staleness diagnostics;
7. if minimum coverage is not met, report consensus unavailable rather than fabricating one through line interpolation.

The exact minimum-book rule is an implementation/data-coverage decision to be pre-registered before scoring.

### Why median rather than mean first

Individual books can differ materially, quotes can be stale, and a single book can temporarily print an outlying price. The median is a conservative robust aggregation operator that limits one-source leverage without requiring historical book-quality weights.

This is primarily **E — engineering/design inference**, not a claim that research proves median is NFL-optimal.

## BASELINE — consensus handicap/total line summary

When the goal is to summarize the market's central quoted spread/total rather than estimate probability at one common threshold:

- report the median quoted line across eligible books;
- preserve the paired prices separately;
- do not call the median line the market expected mean margin/total.

This is a descriptive market benchmark, not a probabilistic estimand identity.

## BASELINE — target-book exclusion for edge benchmarking

When evaluating Ball Knower against an executable quote from book `B`, compute a secondary **leave-one-book-out reference consensus** excluding `B` when sufficient other books remain.

Reason: including the same executable quote inside the benchmark mechanically pulls the benchmark toward the price being evaluated and obscures whether Ball Knower disagrees with the rest of the market.

The full-market consensus may still be stored descriptively.

## Vig-removal methods

### BASELINE

Multiplicative normalization for two-way spread/total/moneyline snapshots because it is transparent, stable and requires no additional fitted assumptions.

### TEST

- Shin probabilities;
- power method;
- additive methods where valid;
- empirically calibrated favorite/longshot corrections.

Štrumbelj (2014) found Shin probabilities more accurate than basic normalization across a broad multi-sport dataset, so Shin is a required challenger. That study does not prove NFL-specific superiority, especially for liquid two-way markets.

## Aggregation/weighting

### BASELINE

Equal book weights + median aggregation.

### TEST

- arithmetic mean;
- geometric/logit pooling;
- liquidity/reliability weights;
- historically learned book weights;
- exchange-inclusive weighting;
- source-specific bias calibration.

Any learned weight uses prior-time data only.

## Different-line consensus

### BASELINE

Do not infer a same-threshold probability if books do not quote the same threshold in adequate numbers.

### TEST

- local interpolation using alternate lines from the same book;
- fitted market-implied CDF;
- distributional reconstruction from multiple spread/total alt lines;
- line/price joint pooling across books.

This is deliberately conservative because interpolation silently adds another model to what is supposed to be a market benchmark.

## Required diagnostics

Store/report at minimum:

- as-of timestamp;
- contributing books;
- quote timestamps/staleness where available;
- common threshold;
- per-book raw prices;
- per-book no-vig probabilities;
- median consensus probability;
- cross-book dispersion;
- target-book-excluded consensus when applicable;
- reason code if consensus unavailable.

## Promotion experiment

The reference-market recipe itself should be benchmarked chronologically using realized outcomes, but not optimized solely for betting ROI.

Compare candidate recipes with:

- Brier/log score at identical thresholds;
- calibration/reliability;
- stability across seasons;
- sensitivity to book coverage;
- resistance to stale/outlying quotes.

A learned weighting/vig-removal/interpolation recipe can be promoted only on prior-time development data and must face the same untouched promotion gate discipline as model choices.

## Final classification

### LOCK

- reference market is separate from executable quote;
- contemporaneous timestamp discipline;
- no averaging probabilities across different thresholds without an explicit model;
- preserve book identity and source timestamps;
- market benchmark cannot contaminate the structural football forecast.

### BASELINE

- same-threshold quotes only;
- per-book multiplicative no-vig normalization;
- equal-weight median probability consensus;
- median quoted line as descriptive line consensus;
- leave-one-book-out consensus when benchmarking against that book and sufficient coverage remains.

### TEST

- Shin/power vig removal;
- learned/reliability book weights;
- mean/geometric/logit pooling;
- interpolation across differing lines;
- market-implied CDF reconstruction;
- exchange/liquidity weighting.

### DEFER

- hand-selected 'sharp book' truth labels without chronological evidence;
- silently treating one sportsbook's line as the consensus market.

## Evidence classes

- **A/B:** literature on bookmaker odds as probability forecasts, probability aggregation and market information aggregation.
- **A:** Štrumbelj (2014) on vig removal/bookmaker differences across sports.
- **E:** exact same-threshold median/equal-weight Ball Knower baseline.

## Key sources

- Štrumbelj, E. (2014), *On determining probability forecasts from betting odds*, International Journal of Forecasting.
- sports forecasting literature showing betting odds are strong outcome forecasts and aggregated odds can reduce source-specific inefficiency.
- general forecast aggregation literature supporting pooling diverse probabilistic judgments while requiring empirical comparison of aggregation operators.