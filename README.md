# ml-labs

What I'm doing to get ML engineering experience by building things. I study
Electronic Engineering at ITA (graduating 2028), so the math is there: linear
algebra, calculus, probability, signals. The goal now is to train, evaluate,
and ship models whose results I can defend.

Every project ends with a reproducible experiment: the baseline, the split,
the metrics, the errors, and what did or did not improve. A model that loses
to its baseline is still a finished experiment when I can explain the result.

## Priority

One active milestone at a time. First understand a training loop, then take
a small prediction problem all the way into a running service. Add text and
ranking after that. Large training runs come later.

| Order | Project | Concepts | First finish line |
|---|---|---|---|
| 1 | **gradlab**: autograd from scratch | Chain rule, gradient checks, loss, optimization | Train an MLP on a small 2D dataset and explain its learning curves |
| 2 | **demandlab**: bicycle demand forecasting | Tabular data, regression, temporal validation, feature availability, model operations | Reproduce a forecast experiment and serve the selected baseline or model |
| 3 | **textdesk**: route forum posts by topic | Text features, classification, calibration, abstention, error analysis | Route confident predictions and measure the trade-off with sending uncertain ones for review |
| 4 | **linerhound**: music recommendation | Entity matching, retrieval, ranking, implicit feedback, larger datasets | Evaluate recommendations against popularity and serve a reproducible model |
| Later | **tinygpt** and **finetune** | Neural-network training systems and model adaptation | Bounded experiments after the first deployed models |

Only gradlab has implementation in this repository today. Demandlab and
textdesk have plans; linerhound is planned in its own repository. Later
projects are optional extensions, not prerequisites for finishing this path.

### 1. gradlab — finish the first training loop

The [existing plan](gradlab/docs/PLAN.md) stays intact. Finish step 2 first:
scalar ops, Neuron, Layer, MLP, and gradient descent on the two-spirals dataset.
Check gradients numerically, reset them correctly, fix the random seed, and
record loss and accuracy on separate training and evaluation points.

**Stop for this milestone:** explain why the weights change, show that the
loss falls, and diagnose a deliberately bad learning rate. Then start
demandlab. Return to tensors, MNIST, and the PyTorch comparison after the
first applied model has been served.

### 2. demandlab — a complete tabular ML project

Forecast the next hour's total bicycle rentals using the public
[UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
Start with calendar information and past demand; add only features actually
available at the forecast time. The dataset contains historical hourly
counts, so serving and monitoring use a chronological replay, not a live feed.

Learn data inspection, missing-hour handling, leakage, chronological splits,
linear regression, tree ensembles, feature engineering, and error analysis.
Compare a seasonal baseline, a regularized linear model, and a boosted-tree
model. Keep the final test period out of model selection.

Deployment is part of this project: a versioned preprocessing/model artifact,
a small API, replayed predictions, delayed labels, basic monitoring, and a
tested rollback. A retraining experiment must pass fixed validation gates
before replacing the current artifact.

[Step-by-step plan](demandlab/docs/PLAN.md).

### 3. textdesk — predictions that can ask for review

Build a topic router using
[20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset).
Start with a majority-class baseline, then TF-IDF and logistic regression.
Strip headers, signatures, and quoted replies; inspect duplicates and the
remaining shortcuts before trusting a score.

Measure per-class errors, macro-F1, probability quality, and how many posts
can be routed at a chosen precision. Fit calibration and choose the review
threshold on held-out development data. An uncertain prediction returns
`needs_review`; it does not silently become the most likely label.

Compare embeddings only after the sparse baseline and evaluation work.
Reuse demandlab's small serving pattern, adding model version, confidence,
and the reason a post was held. The public dataset is a benchmark: report
its limits and test a separately labeled sample before claiming performance
on another kind of text.

[Step-by-step plan](textdesk/docs/PLAN.md).

### 4. linerhound — matching, retrieval, and ranking

Keep the existing music project in its own repository. Before increasing
model complexity, define what counts as the same recording: a live version,
a remaster, and a cover need an explicit policy. Inspect a small labeled
matching set; treat synthetic title corruptions as stress tests, not proof
of real-world matching quality.

Compare popularity, co-occurrence, and ALS or item2vec on future listening
events. Keep only history available at recommendation time. Report Recall@K
and NDCG@K, candidate-set construction, and results for sparse-history users
and less popular tracks. Offline gains do not establish user satisfaction.

Serve the first useful recommender before building a custom index. Start
with exact retrieval, measure latency and memory, and add approximate search
when the measurements justify it. DuckDB over Parquet keeps dump processing
within the laptop's memory budget. Scheduled ingestion, retraining, and model
promotion extend the lifecycle already practiced in demandlab.

### Later — deeper neural-network experiments

- **gradlab steps 3–5:** tensor autograd, MNIST, and a PyTorch comparison.
- **tinygpt:** a small decoder-only transformer on TinyStories. Begin with a
  short memory/throughput probe, then set a training budget. Learn AdamW,
  scheduling, mixed precision, checkpoint/resume, and KV-cache inference.
  A custom BPE tokenizer and scaling experiments are separate extensions.
- **finetune:** canonicalize messy music titles. Define the target policy,
  labeled evaluation set, and scorer first; compare rules, a prompted model,
  and a small adapted model on quality, cost, and latency. Use an existing
  LoRA implementation for the first experiment; implementing LoRA is a later
  exercise. Establish VRAM requirements with a short run before committing
  to a model size.

## How it's run

Concept before code. I write the experiment logic: target definition, data
split, features, metrics, baselines, and interpretation. For gradlab I also
write the engine, layers, loss, and training loop. Claude can help with
scaffolding, tests, and review; I must be able to explain every result.

Use established model implementations in the applied projects. Implement a
component from scratch when understanding that component is the milestone.
Python with `uv`; dependencies are added when a step needs them. Each step
is one commit that runs end to end, with a dated entry in its project's log.

Every experiment records the data source/license and checksum, split dates
or IDs, seed, code revision, preprocessing, parameters, baseline, metrics,
and inspected failures. Fit preprocessing only on training data. Tune on
development data and open the final test set after choices are fixed.
Synthetic fixtures test the code; claims about model quality use real labels.

## Machine

RTX 4060 Laptop (8 GB VRAM), 12 cores, 16 GB RAM. The first applied projects
are designed for CPU runs on bounded datasets. Neural-network extensions
start with a measured memory and runtime budget. Data and trained artifacts
stay out of Git; commit their manifests and reproduction instructions.

## Reading, when a project asks for it

- Karpathy, *Neural Networks: Zero to Hero*: alongside gradlab; implement a
  small attempt before watching its solution.
- [Inria's scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/):
  preprocessing, regression, trees, and model evaluation for demandlab.
- [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
  and [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html):
  read before the first scored experiment.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html):
  before setting textdesk's review threshold.
- Chip Huyen, *Designing Machine Learning Systems*: as deployment and
  monitoring questions arise in demandlab and linerhound.
