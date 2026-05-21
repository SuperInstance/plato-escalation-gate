"""Extended tests for plato_escalation_gate — EscalationGate training & edge cases."""

import pytest
import torch
import numpy as np
from plato_escalation_gate import EscalationGate, generate_escalation_data


class TestEscalationGateExtended:
    def test_param_count(self):
        gate = EscalationGate()
        assert gate.param_count == 737

    def test_size_bytes(self):
        gate = EscalationGate()
        assert gate.size_bytes == 737 * 4

    def test_forward_shape(self):
        gate = EscalationGate()
        x = torch.randn(10, 5)
        out = gate(x)
        assert out.shape == (10, 1)

    def test_forward_output_range(self):
        gate = EscalationGate()
        x = torch.randn(10, 5)
        out = gate(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_should_escalate(self):
        gate = EscalationGate()
        x = torch.randn(5, 5)
        result = gate.should_escalate(x)
        assert result.shape == (5,)
        assert result.dtype == torch.bool

    def test_should_escalate_threshold(self):
        gate = EscalationGate()
        # Very low confidence + high anomaly should escalate
        x = torch.FloatTensor([[0.1, 5, 0.3, 0.9, 15.0]])
        result = gate.should_escalate(x, threshold=0.5)
        assert result.shape == (1,)

    def test_single_sample(self):
        gate = EscalationGate()
        x = torch.randn(1, 5)
        out = gate(x)
        assert out.shape == (1, 1)


class TestGenerateEscalationData:
    def test_shapes(self):
        X, y = generate_escalation_data(n=100)
        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_labels_binary(self):
        X, y = generate_escalation_data(n=500)
        assert set(y.numpy().flatten()).issubset({0.0, 1.0})

    def test_custom_noise(self):
        X, y = generate_escalation_data(n=100, noise_rate=0.0)
        # With no noise, labels should follow the rule exactly
        for i in range(100):
            conf = X[i, 0].item()
            dr = X[i, 2].item()
            anom = X[i, 3].item()
            expected = (conf < 0.4 and dr > 0.15) or anom > 0.8
            assert y[i].item() == float(expected)

    def test_default_params(self):
        X, y = generate_escalation_data()
        assert X.shape[0] == 3000
