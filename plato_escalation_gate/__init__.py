"""
plato-escalation-gate — Tiny Escalation Decision Gate

A 737-parameter binary classifier (4KB) that decides whether a situation
requires escalation to a higher-level agent or model. Small enough for
WASM deployment and edge inference.

Usage:
    from plato_escalation_gate import EscalationGate

    gate = EscalationGate()
    decision = gate(torch.FloatTensor([[0.3, 5, 0.12, 0.2, 15.0]]))
    # decision > 0.5 → escalate
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["EscalationGate", "generate_escalation_data"]


class EscalationGate(nn.Module):
    """Binary escalation decision gate.

    Architecture: Linear(5→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid

    Total parameters: 737 (2,948 bytes at float32)

    Inputs (5 features):
        0. confidence — agent confidence score [0, 1]
        1. activity   — Poisson-distributed activity count
        2. drift_rate — exponential drift rate
        3. anomaly    — anomaly score [0, 1]
        4. time_since — time since last check (exponential scale)

    Output:
        Sigmoid probability that the situation requires escalation.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return escalation probability for each input sample.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 5) — [confidence, activity, drift_rate, anomaly, time_since]

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — escalation probability per sample.
        """
        return self.net(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_bytes(self) -> int:
        return self.param_count * 4

    def should_escalate(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Boolean escalation decision."""
        return self(x).squeeze(-1) > threshold


def generate_escalation_data(n: int = 3000, noise_rate: float = 0.05):
    """Generate synthetic escalation training data.

    The ground-truth rule: escalate if (confidence < 0.4 AND drift > 0.15)
    OR anomaly > 0.8, with a configurable noise rate for label flips.

    Parameters
    ----------
    n : int
        Number of samples.
    noise_rate : float
        Probability of flipping a label (simulates noisy real data).

    Returns
    -------
    tuple[Tensor, Tensor]
        (X, y) where X is (n, 5) float features, y is (n,) float labels.
    """
    X = np.zeros((n, 5))
    y = np.zeros(n)

    for i in range(n):
        conf = np.random.random()
        dr = np.random.exponential(0.1)
        anom = np.random.random()
        X[i] = [conf, np.random.poisson(5), dr, anom, np.random.exponential(10)]
        esc = (conf < 0.4 and dr > 0.15) or anom > 0.8
        if np.random.random() < noise_rate:
            esc = not esc
        y[i] = int(esc)

    return torch.FloatTensor(X), torch.FloatTensor(y)
