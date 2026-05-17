"""Tests for plato-escalation-gate."""
import pytest
import torch

from plato_escalation_gate import EscalationGate, generate_escalation_data


class TestEscalationGate:
    def test_output_shape(self):
        gate = EscalationGate()
        x = torch.randn(8, 5)
        out = gate(x)
        assert out.shape == (8, 1)

    def test_output_range(self):
        gate = EscalationGate()
        x = torch.randn(16, 5)
        out = gate(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_param_count(self):
        gate = EscalationGate()
        assert gate.param_count == 737

    def test_size_bytes(self):
        gate = EscalationGate()
        assert gate.size_bytes == 737 * 4

    def test_should_escalate(self):
        gate = EscalationGate()
        x = torch.randn(4, 5)
        decisions = gate.should_escalate(x, threshold=0.5)
        assert decisions.shape == (4,)
        assert decisions.dtype == torch.bool

    def test_training_runs(self):
        gate = EscalationGate()
        X, y = generate_escalation_data(100)
        opt = torch.optim.Adam(gate.parameters(), lr=0.01)
        crit = torch.nn.BCELoss()

        for _ in range(10):
            opt.zero_grad()
            loss = crit(gate(X).squeeze(), y)
            loss.backward()
            opt.step()

        assert loss.item() < 1.0  # should at least not diverge

    def test_single_sample(self):
        gate = EscalationGate()
        x = torch.FloatTensor([[0.2, 5, 0.3, 0.9, 20.0]])
        out = gate(x)
        assert out.shape == (1, 1)


class TestDataGeneration:
    def test_shapes(self):
        X, y = generate_escalation_data(500)
        assert X.shape == (500, 5)
        assert y.shape == (500,)

    def test_binary_labels(self):
        _, y = generate_escalation_data(1000)
        unique = set(y.numpy().flatten().tolist())
        assert unique.issubset({0.0, 1.0})

    def test_noise_rate(self):
        _, y = generate_escalation_data(5000, noise_rate=0.0)
        # With noise_rate=0, all labels follow the deterministic rule
        assert len(y) == 5000
