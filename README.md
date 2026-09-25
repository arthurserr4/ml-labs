# ml-labs

What I'm doing to get ML engineering experience by building things, not by
watching courses. I study Electronic Engineering at ITA (graduating 2028), so
the math is there: linear algebra, calculus, probability, signals. What I
don't have is a model I trained and shipped.

An ML engineer's work is mostly data, evaluation, and getting models to run
reliably in production. The modelling itself is maybe a fifth of it. So the
projects below cover all of it, and each one ends with numbers: a baseline,
what beat it, by how much, and what didn't work.

## Projects

In the order I plan to start them. They overlap; I switch by what I feel like
that week.

| # | Project | What it teaches | Where |
|---|---|---|---|
| 1 | **gradlab**: autograd from scratch | What `loss.backward()` does. Backprop, gradient checking, initialization, why training diverges. | `gradlab/` |
| 2 | **linerhound**: music recommender | Data too big for memory, offline evaluation, baselines, classical models (co-occurrence, item2vec, ALS). | own repo, not public yet |
| 3 | **tinygpt**: a small transformer on my GPU | A real training loop: tokenizer, data loading, AdamW, LR schedule, mixed precision, checkpoints, reading loss curves, inference with a KV cache. | `tinygpt/` |
| 4 | **linerhound part 2**: production | Serving under a latency budget, an approximate nearest-neighbour index, daily retraining from new data, validation gates, a model that is promoted only if it beats the current one, drift monitoring. | own repo, not public yet |
| 5 | **finetune**: adapt a small open LLM | Eval set first, LoRA written by hand, fine-tuned small model vs prompted large model on quality, cost and latency. | `finetune/` |

### 1. gradlab

A scalar autograd engine (a `Value` that remembers how it was computed and
can run the chain rule backwards), then a tensor version on numpy, then an MLP
trained on MNIST. Last step: the same network in PyTorch, and the numbers have
to match mine. After this, PyTorch is a faster version of something I've
written, not magic.

### 2. linerhound

Planned in its own repo (not public yet), steps 1 to 7.

### 3. tinygpt

A decoder-only transformer, ~10–30M parameters, trained on TinyStories (small
models produce readable text on it). BPE tokenizer written by me. Training
runs for hours, so it needs checkpoint and resume. Includes a small scaling
experiment: three model sizes, loss against compute. Ends with generation
speed with and without a KV cache.

### 4. linerhound part 2

Added to linerhound's plan when step 7 lands. The model from part 1 behind an
HTTP API with a p99 latency target; an IVF index I write, measured against
exact search for recall and speed; a scheduled job that pulls each new
ListenBrainz dump, checks it, retrains, evaluates, and replaces the served
model only if it wins; a report of how the input data drifts over the weeks.

### 5. finetune

Take a narrow task, for example turning messy track titles ("Song - 2011
Remaster", "Song (feat. X) [Live]") into canonical artist and title, which
linerhound needs anyway. Write the eval set and scorer first. Then: prompt a
large model (the bar), fine-tune a ~0.5B open model with LoRA that I write,
compare quality, cost per 1k items and latency.

## Machine

RTX 4060 Laptop (8 GB VRAM), 12 cores, 16 GB RAM. Enough for everything
above. Constraints this puts on the design: tinygpt stays under ~50M
parameters, finetune uses a small model, linerhound never loads all dumps
into memory at once (DuckDB over Parquet).

## How it's run

Same rules as my Go and Rust labs: one step at a time, concept before code, I
write the core (the models, the losses, the metrics, the training loop),
Claude writes the plumbing and reviews what I write. Python with `uv`. Each
step is one commit that runs end to end, with a dated log entry in the
project's plan.

## Reading, only when a project asks for it

- Andrej Karpathy, *Neural Networks: Zero to Hero* (videos). The
  micrograd and makemore/GPT videos line up with gradlab and tinygpt. Watch
  after writing my version, not before.
- Chip Huyen, *Designing Machine Learning Systems*. For linerhound part 2.
- Martin Zinkevich, *Rules of Machine Learning* (Google). Short. Read before
  linerhound step 3.
