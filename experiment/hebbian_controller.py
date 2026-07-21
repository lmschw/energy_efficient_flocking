"""Hebbian-plasticity MLP controller (paper Section 2.1, Eq. 1).

1:1 port of hebbianStep.m's forward pass + Hebbian weight update, and the per-agent
weight-initialization loop in simulation_free_global_mod_2.m.

Note: evaluateABCD.m / optimizeABCD.m in the MATLAB source implement a different,
stale 14-14-14-2 (1680-parameter) architecture that matches neither hebbianStep.m's
actual 10-10-10-2 (880-parameter) network nor the paper's stated architecture -- this
port follows hebbianStep.m, simulation_free_global_mod_2.m, and the paper, not those
mismatched driver scripts.
"""
import numpy as np

try:
    import config
except ModuleNotFoundError:
    from . import config

# Layer shapes, in flatten/unflatten order (matches hebbianStep.m's W1, W2, W3).
_LAYER_SHAPES = (
    ("1", (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN)),
    ("2", (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_HIDDEN)),
    ("3", (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS)),
)
_LETTERS = ("A", "B", "C", "D")


def init_weights():
    """Fresh, randomly-initialized NN weights for one agent -- not evolved, re-drawn
    every episode. Paper: "uniform distribution in [-1, 1]" for all three matrices
    (the MATLAB source samples W1 from randn() instead, which we treat as a bug)."""
    r = config.HEBBIAN_WEIGHT_INIT_RANGE
    w1 = np.random.uniform(-r, r, (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN))
    w2 = np.random.uniform(-r, r, (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_HIDDEN))
    w3 = np.random.uniform(-r, r, (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS))
    return w1, w2, w3


def unflatten_abcd(flat):
    """Maps an 880-length genome vector to a dict of 12 matrices -- A1, A2, A3, B1,
    B2, B3, C1, C2, C3, D1, D2, D3 -- matching hebbianStep.m's R.A1..R.D3 fields."""
    flat = np.asarray(flat, dtype=float)
    rules = {}
    idx = 0
    for letter in _LETTERS:
        for suffix, shape in _LAYER_SHAPES:
            size = shape[0] * shape[1]
            rules[letter + suffix] = flat[idx:idx + size].reshape(shape)
            idx += size
    assert idx == config.HEBBIAN_N_ABCD, f"expected to consume {config.HEBBIAN_N_ABCD}, got {idx}"
    return rules


def _normalize(w):
    """Mirrors hebbianStep.m's normalize(): scale down only if a weight exceeds 1 in
    magnitude, to prevent unbounded growth (paper: "we keep normalising the weights")."""
    maxval = np.max(np.abs(w))
    return w / maxval if maxval > 1.0 else w


def hebbian_step(x_in, w1, w2, w3, rules):
    """One forward pass + Hebbian update for a single agent.

    x_in: (N_INPUTS,) sensory vector. w1/w2/w3: this agent's current NN weights.
    rules: dict from unflatten_abcd(), shared across every agent in the swarm.
    Returns (v, w, w1_new, w2_new, w3_new).
    """
    x_in = x_in.reshape(-1, 1)               # (10, 1) column vector, matches MATLAB's `in`
    h1 = np.maximum(0.0, x_in.T @ w1)         # (1, 10)
    h2 = np.maximum(0.0, h1 @ w2)             # (1, 10)
    out = np.tanh(h2 @ w3)                    # (1, 2)

    v = out[0, 0] * config.HEBBIAN_LINEAR_VEL_MAX
    w = out[0, 1] * config.HEBBIAN_ANGULAR_VEL_MAX

    eta = config.HEBBIAN_LEARNING_RATE
    # Broadcasting (10,1)*(10,10) and (1,10)*(10,10) below reproduces MATLAB's
    # repmat(in,1,10) / repmat(h1,10,1) without an explicit repeat.
    w1_new = _normalize(w1 + eta * (rules["A1"] * (x_in @ h1) + rules["B1"] * x_in
                                     + rules["C1"] * h1 + rules["D1"]))
    w2_new = _normalize(w2 + eta * (rules["A2"] * (h1.T @ h2) + rules["B2"] * h1.T
                                     + rules["C2"] * h2 + rules["D2"]))
    w3_new = _normalize(w3 + eta * (rules["A3"] * (h2.T @ out) + rules["B3"] * h2.T
                                     + rules["C3"] * out + rules["D3"]))
    return v, w, w1_new, w2_new, w3_new
