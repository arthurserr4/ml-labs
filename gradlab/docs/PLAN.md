# Plan — gradlab, autograd from scratch

Write the thing `loss.backward()` does, train a network with it, then check
it against PyTorch. When this is done, PyTorch should look like a faster
version of something I wrote.

## How this is run

One step at a time, concept before code. I write the engine, the layers, the
loss and the training loop. Claude writes the scaffolding, the tests
(gradient checks), data download, and reviews what I write.

## Steps

Each step is a commit and runs end to end before the next one starts.

0. **Scaffold and plan.** `uv` project, pytest, this file.
1. **Scalar `Value` with `backward()`.** `+`, `*`, `tanh`, and a backward
   pass in reverse topological order. Tested against finite differences:
   `(f(x+ε) − f(x−ε)) / 2ε` has to match my gradient for every input.
   Concepts first: reverse-mode autodiff, topological order, why gradients
   accumulate (`+=`).
2. **Enough ops to train something.** `-`, `/`, `**`, `exp`, `relu`,
   mixing with plain floats (`2 * v`). A `Neuron`, `Layer` and `MLP`. Train
   on a small 2D dataset I generate (two interleaved spirals) with plain
   gradient descent; print loss and accuracy. Concepts first: loss
   functions, gradient descent, why forgetting to zero the gradients breaks
   training.
3. **Tensors on numpy.** Same idea, but each node holds an array: `@`,
   broadcasting `+`, `sum`, `relu`, softmax cross-entropy. Same gradient
   check, now per element. Concepts first: the backward of a matrix
   product, why a broadcast in the forward pass is a sum in the backward
   pass, a numerically stable log-softmax.
4. **MNIST.** An MLP trained with minibatches, target ~97% test accuracy.
   Experiments with numbers in the log: loss at initialization (should be
   ln 10 ≈ 2.30), bad initialization vs He initialization, learning rate
   too high and too low. Concepts first: initialization and activation
   scale, minibatch noise, train vs test accuracy.
5. **Against PyTorch.** The same MLP in PyTorch, starting from the same
   weights. Loss and gradients have to match mine to ~1e-6 after one step.
   Then time one epoch: mine on numpy, PyTorch on CPU, PyTorch on the GPU.

## Decisions already taken

- **Scalar first, tensors second.** The scalar version is slow but every
  gradient is one line I can check by hand. The tensor version is the same
  algorithm with shapes.
- **Gradient checks are the tests.** A backward pass that runs without
  errors can still be wrong. Finite differences don't lie (much).
- **Dependencies one at a time, each with a reason here.** Current list:
  `pytest` (dev). Next: `numpy` (step 3, arrays), `torch` (step 5, the
  comparison).
- **`data/` is never committed.** MNIST is downloaded by a script.

---

# Log

- **2026-09-25 — step 0.** `uv` project with pytest, this plan. The
  gradient check for step 1 is written but lands with step 1, since it
  fails until `Value` exists.
