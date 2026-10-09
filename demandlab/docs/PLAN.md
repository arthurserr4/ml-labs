# Plan — demandlab, bicycle demand forecasting

Train a next-hour demand predictor and operate it on a replay of historical
data. Learn the whole path from a table to an evaluated, versioned prediction
service.

## Prediction contract

At the end of hour `t`, predict total rentals during hour `t+1`. Assume hour
`t` has been fully observed; document that availability assumption. One row
is one forecast origin. Features may use calendar information for `t+1` and
observations through `t`, never outcomes from `t+1`.

Use `hour.csv` from the
[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
with its attribution and license recorded in the data manifest. Start with
calendar features and past total demand. If weather is added, use past
observations; future observed weather is not a forecast input. Exclude the
target hour's `casual` and `registered` counts: together they reveal `cnt`.

## Steps

Each step starts with its concept, then implementation, checks, and a dated
log entry. Do not implement every step in one pass.

0. **Data contract and inspection.** Create a `uv` project and reproducible
   download. Record source, license, checksum, schema, date coverage, missing
   timestamps, and duplicate timestamps. Build a complete hourly index to
   expose gaps. A missing hour is unknown, not automatically zero. Join the
   next-hour label by timestamp so a missing row cannot change the horizon.
1. **An honest baseline.** Freeze chronological train, development, and final
   test periods. Start with last-hour and same-hour-last-week predictions,
   with a documented training-only fallback for missing history. Report MAE,
   RMSE, and error by hour and working-day status. Add a check that every
   feature's observation time is at or before the forecast origin.
2. **First learned model.** A scikit-learn pipeline with calendar encoding,
   lagged demand, and regularized linear regression. Fit transforms on the
   training period only. Explain coefficients, residuals, and failure cases.
   Rolling features must be shifted to respect the prediction contract.
3. **Nonlinear model.** Compare a boosted-tree regressor with the same splits
   and available information. Bound the parameter search in advance. Use
   rolling-origin development folds, keeping final test data untouched.
   Run feature ablations and inspect large errors before adding features.
4. **Freeze the experiment.** Select the baseline or model using development
   results, then evaluate once on the final period. Report absolute and
   relative differences, error slices, training time, inference latency,
   and variation across development folds. A loss to the baseline is a
   valid outcome; do not repeatedly tune against the final period.
5. **Serve it.** Save preprocessing and model together with a manifest.
   Expose a validated request/response through a small FastAPI application.
   Return prediction, model version, and forecast origin. Verify that batch
   and API predictions match. Measure latency on a stated machine and load.
6. **Replay operations.** Replay an operational exercise chronologically,
   releasing labels only when their hours finish. Track missing inputs,
   input shifts, prediction distribution, and error after labels arrive.
   Train candidates on past data and evaluate on a later promotion window;
   require fixed quality/schema checks before swapping artifacts. Test a
   rejected candidate and rollback. Reusing benchmark data here demonstrates
   lifecycle mechanics, not an additional independent quality result.

## Finish line

A documented command reproduces the experiment; another starts the service.
The report names the selected baseline or model and explains its limitations.
A replay demonstrates label delay, monitoring, a promotion decision, and
rollback. No dashboard, GPU, or new infrastructure platform is required.

## Boundaries

- The historical dataset cannot demonstrate present-day live performance.
- No random row split for a claim about future demand.
- Define nonnegative prediction handling before evaluating models.
- Input drift is a reason to investigate, not proof that retraining helps.
- Inspect both metrics and errors; keep the raw data immutable.
- Use libraries for the regressors. The learning work is in framing, data,
  evaluation, and operating the result.

## Log

- **2026-10-09 — step 0, setup.** Added a dependency-free `uv` project, a
  checksum-pinned downloader, the source/license manifest, and a small CSV
  inspection script. The downloaded hourly file has 17,379 rows and 17
  columns. No features, labels, splits, or models have been implemented.
  Next: inspect timestamp coverage, duplicates, and missing hours before
  constructing next-hour examples. Step 0's data audit is still open.
