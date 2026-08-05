"""Hebbian-plasticity MLP controller -- ported verbatim (only the config import changed)
from the verified/tested experiment/hebbian_controller.py in the main simulation
package. See that file for the derivation notes and MATLAB reference (hebbianStep.m).
"""
import numpy as np

import controller_config as cfg

_LAYER_SHAPES = (
    ("1", (cfg.N_INPUTS, cfg.N_HIDDEN)),
    ("2", (cfg.N_HIDDEN, cfg.N_HIDDEN)),
    ("3", (cfg.N_HIDDEN, cfg.N_OUTPUTS)),
)
_LETTERS = ("A", "B", "C", "D")


def init_weights():
    """Fresh, randomly-initialized NN weights -- not evolved, re-drawn once at the start
    of each real deployment run (there is no equivalent of "restarting an episode" on
    hardware; this just seeds the weights the Hebbian rule will continuously adapt for
    as long as the experiment runs)."""
    r = cfg.WEIGHT_INIT_RANGE
    w1 = np.random.uniform(-r, r, (cfg.N_INPUTS, cfg.N_HIDDEN))
    w2 = np.random.uniform(-r, r, (cfg.N_HIDDEN, cfg.N_HIDDEN))
    w3 = np.random.uniform(-r, r, (cfg.N_HIDDEN, cfg.N_OUTPUTS))
    return w1, w2, w3


def unflatten_abcd(flat):
    """Maps an 880-length genome vector (as saved by optimize_hebbian.py) to a dict of
    12 matrices: A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3."""
    flat = np.asarray(flat, dtype=float)
    rules = {}
    idx = 0
    for letter in _LETTERS:
        for suffix, shape in _LAYER_SHAPES:
            size = shape[0] * shape[1]
            rules[letter + suffix] = flat[idx:idx + size].reshape(shape)
            idx += size
    assert idx == cfg.N_ABCD, f"expected to consume {cfg.N_ABCD}, got {idx}"
    return rules


def _normalize(w):
    maxval = np.max(np.abs(w))
    return w / maxval if maxval > 1.0 else w


def hebbian_step(x_in, w1, w2, w3, rules):
    """One forward pass + Hebbian update. x_in: (N_INPUTS,) sensory vector. w1/w2/w3:
    this robot's current NN weights. rules: dict from unflatten_abcd(). Returns
    (v, w, w1_new, w2_new, w3_new)."""
    x_in = x_in.reshape(-1, 1)
    h1 = np.maximum(0.0, x_in.T @ w1)
    h2 = np.maximum(0.0, h1 @ w2)
    out = np.tanh(h2 @ w3)

    v = out[0, 0] * cfg.LINEAR_VEL_MAX
    w = out[0, 1] * cfg.ANGULAR_VEL_MAX

    eta = cfg.LEARNING_RATE
    w1_new = _normalize(w1 + eta * (rules["A1"] * (x_in @ h1) + rules["B1"] * x_in
                                     + rules["C1"] * h1 + rules["D1"]))
    w2_new = _normalize(w2 + eta * (rules["A2"] * (h1.T @ h2) + rules["B2"] * h1.T
                                     + rules["C2"] * h2 + rules["D2"]))
    w3_new = _normalize(w3 + eta * (rules["A3"] * (h2.T @ out) + rules["B3"] * h2.T
                                     + rules["C3"] * out + rules["D3"]))
    return v, w, w1_new, w2_new, w3_new
