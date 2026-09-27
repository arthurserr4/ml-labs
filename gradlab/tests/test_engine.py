"""Gradient checks: backward() must agree with finite differences.

For each case, the gradient from backward() is compared with the central
difference (f(x+eps) - f(x-eps)) / 2eps, one input at a time. Its error is
O(eps^2) plus float rounding (~1e-16 / eps), so with eps = 1e-6 anything
further apart than 1e-6 is a bug, not noise.
"""

import math

import pytest

from gradlab.engine import Value

EPS = 1e-6
TOL = 1e-6


def numeric_grads(f, xs):
    grads = []
    for i in range(len(xs)):
        up = list(xs)
        up[i] += EPS
        down = list(xs)
        down[i] -= EPS
        f_up = f(*[Value(x) for x in up]).data
        f_down = f(*[Value(x) for x in down]).data
        grads.append((f_up - f_down) / (2 * EPS))
    return grads


def analytic_grads(f, xs):
    vs = [Value(x) for x in xs]
    f(*vs).backward()
    return [v.grad for v in vs]


CASES = {
    "add": (lambda a, b: a + b, [2.0, -3.0]),
    "mul": (lambda a, b: a * b, [2.0, -3.0]),
    "tanh": (lambda a: a.tanh(), [0.7]),
    "neuron": (lambda x, w, b: (x * w + b).tanh(), [0.5, -1.2, 0.3]),
    # `a` is used three times: its gradient is the sum of all three paths.
    "reuse": (lambda a: a * a + a, [1.5]),
    # Two paths from a and b to the output that meet again at the end.
    "diamond": (lambda a, b: (a * b).tanh() * (a + b), [0.4, -0.9]),
}


@pytest.mark.parametrize("name", CASES)
def test_backward_matches_finite_differences(name):
    f, xs = CASES[name]
    for got, want in zip(analytic_grads(f, xs), numeric_grads(f, xs)):
        assert got == pytest.approx(want, abs=TOL)


def test_forward_values():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert a.tanh().data == pytest.approx(math.tanh(2.0))


def test_output_grad_is_one():
    out = Value(2.0) * Value(3.0)
    out.backward()
    assert out.grad == 1.0


def test_backward_does_not_touch_unrelated_values():
    a, b, unused = Value(2.0), Value(3.0), Value(5.0)
    (a * b).backward()
    assert unused.grad == 0.0
