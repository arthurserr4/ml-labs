# demandlab

Learn to predict next-hour bicycle rentals, evaluate on later observations,
and eventually serve a trained model. The [roadmap](docs/PLAN.md) describes
the full experiment.

## Start with the raw table

From `demandlab/`, download the dataset and inspect it:

```bash
uv run python scripts/download_data.py
uv run python scripts/inspect_data.py
```

Open [the inspection script](scripts/inspect_data.py): it uses Python's
standard CSV reader to count rows, list columns, and show five observations.
The raw file is `data/raw/hour.csv`.

Each raw row describes an observed hour. `dteday` is the date, `hr` is the
hour, and `cnt` is that hour's total rentals. For example, a row marked
`13` describes rentals during 13:00–14:00.

At 14:00, assume the 13:00–14:00 observation is available. We want to predict
rentals during 14:00–15:00. Past rental counts can be inputs; the next hour's
count is the target, which becomes available after that hour ends.

## Your next task

Before fitting a model, write the timestamp inspection code:

1. Combine `dteday` and `hr` into an hourly timestamp.
2. Report the earliest and latest timestamps and count duplicate timestamps.
3. Compare the observations with a complete hourly timeline to identify gaps.

A missing hour means an unknown observation, not necessarily zero rentals.
When you later assemble training examples, find the next hour by timestamp:
the next CSV row may be more than one hour away.

You will write the feature and target construction, chronological splits,
baseline, and model experiment. The current scripts prepare and inspect the
source data; they do not train a predictor.
