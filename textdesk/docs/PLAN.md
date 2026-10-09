# Plan — textdesk, a topic router that can ask for review

Classify a forum post by topic, estimate how reliable the prediction is,
and hold uncertain posts for review. Learn text features, classification,
calibration, and the cost of automatic decisions.

## Task and data

Start with
[20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset).
The input is post text; the target is its newsgroup category. Record the
dataset source, usage terms, loader options, and snapshot checksum.

Load with `remove=('headers', 'footers', 'quotes')`. Inspect examples after
cleaning: the loader's removal is heuristic, not a guarantee against leakage.
Keep the provided test partition locked. Audit duplicate and near-duplicate
posts across partitions; document exclusions and their effect. Group related
posts when splitting development data so copies do not teach the answer.

## Steps

Concept first, then code, checks, and a dated log entry. Reuse the serving
pattern learned in demandlab.

0. **Dataset audit.** Inspect class counts, lengths, empty cleaned posts,
   duplicates, and confusing topic pairs. Write the label policy and reserve
   development subsets for selection, calibration, and threshold choice.
   Keep a manifest of sample IDs and split roles.
1. **Baselines.** Compare majority-class prediction, TF-IDF with naive Bayes,
   and TF-IDF with logistic regression. Fit the vocabulary and IDF only on
   training data, inside a pipeline. Report macro-F1, per-class precision and
   recall, and a confusion matrix on development data.
2. **Error analysis.** Read a fixed sample of mistakes. Separate ambiguous
   labels, missing context, misleading vocabulary, and obvious metadata
   shortcuts. Test a small number of changes, such as word versus character
   n-grams. Keep the same evaluation protocol across experiments.
3. **Probability quality.** Compare raw and calibrated probabilities using
   log loss and reliability diagrams. Fit calibration on data not used to
   fit the classifier or select its hyperparameters. Check sample counts
   behind each confidence bin; a high score is not proof of certainty.
4. **Selective routing.** Choose confidence thresholds on the reserved
   threshold-selection subset. Below threshold, return `needs_review`.
   Report precision among automatically routed posts versus coverage, plus
   per-class results and review volume. Choose an explicit target precision
   and report whether it is achieved; low coverage is a result, not a reason
   to quietly lower the target.
5. **Optional representation experiment.** Compare a fixed pretrained text
   embedding plus a small classifier against the sparse baseline. Measure
   quality, memory, and inference latency. Reuse the clean split protocol;
   document the embedding model and uncertainty about its pretraining data.
   A neural model is optional if the simpler system is adequate.
6. **Final evaluation and service.** Freeze choices, evaluate on the test
   partition, and expose a text-in/result-out API. Return model version,
   decision (`routed` or `needs_review`), predicted topic, and probability.
   Define empty/oversized-input behavior and test artifact reload. A labeled
   sample from another source is a separate transfer test; report its score
   separately from the benchmark.

## Finish line

A reproducible baseline comparison, an inspected error report, and a
precision/coverage curve support the routing threshold. The service exposes
uncertainty and can restore a previous model artifact. Explain which posts
it cannot route reliably and what additional labels would help.

## Boundaries

- Topic labels do not establish urgency, intent, or importance.
- Confidence thresholds do not guarantee detection of unfamiliar topics.
  Evaluate unrelated texts as a separate stress test.
- Human corrections collected only from uncertain posts form a biased
  sample. Audit some confidently routed posts too; keep later evaluation
  examples out of retraining.
- Keep raw post text out of service logs; record IDs and aggregate metrics.
- Synthetic posts test API behavior, not benchmark generalization.
- Start with CPU models and established implementations; train larger
  models only to answer a specific question left by the baseline.

## Log

Planning only. Add a dated entry after each implemented step with the
experiment, result, mistakes, and next question.
