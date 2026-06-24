"""Tests for the clamping summary in sweep.py.

Ensures that the clamping summary does not crash on tensor-valued
BinaryGenerativeParameters and produces correctly deduplicated output.

The clamping summary collects unique eta/gamma/lambdah values from
BinaryGenerativeParameters instances that triggered numerical clamping.
Because BinarySweepParameters.__iter__ yields 0-d torch.Tensor values
for eta/gamma/lambdah (not Python floats), the summary must convert
these to Python scalars via .item() to avoid:
1. TypeError on older PyTorch where tensors are unhashable.
2. Incorrect deduplication on newer PyTorch where tensor hashing is
   identity-based, not value-based.
"""

import torch
from gnm.fitting.experiment_dataclasses import Experiment, RunConfig
from gnm import BinaryGenerativeParameters
from gnm.generative_rules import MatchingIndex


def _make_mock_experiment(eta, gamma, lambdah, clamp_count=1):
    """Create a mock Experiment with tensor-valued binary parameters.

    Mimics what BinarySweepParameters.__iter__ produces — individual elements
    yielded from iterating over ``torch.Tensor`` parameter lists are 0-d
    tensors, not Python floats.
    """
    bp = BinaryGenerativeParameters(
        eta=torch.tensor(eta),
        gamma=torch.tensor(gamma),
        lambdah=torch.tensor(lambdah),
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(),
        num_iterations=100,
    )
    rc = RunConfig(binary_parameters=bp, weighted_parameters=None)
    exp = Experiment(
        run_config=rc,
        evaluation_results={},
        model=None,
        run_history=None,
        clamp_count=clamp_count,
    )
    return exp


def _clamp_summary(clamped_configs):
    """Reproduce the clamping summary pattern from sweep.py.

    This function mirrors the logic in ``perform_grid_sweep()`` and must
    use ``.item()`` to convert 0-d tensors to Python scalars so the
    ``set()`` call works correctly on all PyTorch versions.
    """
    eta_vals = sorted(set(c.eta.item() for c in clamped_configs))
    gamma_vals = sorted(set(c.gamma.item() for c in clamped_configs))
    lambdah_vals = sorted(set(c.lambdah.item() for c in clamped_configs))
    return eta_vals, gamma_vals, lambdah_vals


class TestClampSummary:
    """Verify the clamping summary correctly deduplicates tensor-valued
    BinaryGenerativeParameters attributes."""

    def test_duplicate_values_are_deduplicated(self):
        """Multiple configs with the same eta/gamma/lambdah should collapse
        to a single entry each."""
        experiments = [
            _make_mock_experiment(eta=-5.0, gamma=-1.0, lambdah=0.0),
            _make_mock_experiment(eta=-5.0, gamma=-1.0, lambdah=0.0),
            _make_mock_experiment(eta=-5.0, gamma=-1.0, lambdah=0.0),
        ]
        clamped = [e.run_config.binary_parameters for e in experiments]

        eta_vals, gamma_vals, lambdah_vals = _clamp_summary(clamped)

        assert eta_vals == [-5.0], f"Expected [-5.0], got {eta_vals}"
        assert gamma_vals == [-1.0], f"Expected [-1.0], got {gamma_vals}"
        assert lambdah_vals == [0.0], f"Expected [0.0], got {lambdah_vals}"

    def test_distinct_values_remain_distinct(self):
        """Distinct parameter values should all appear, sorted ascending."""
        experiments = [
            _make_mock_experiment(eta=-5.0, gamma=-1.0, lambdah=0.0),
            _make_mock_experiment(eta=-3.0, gamma=-0.5, lambdah=0.0),
            _make_mock_experiment(eta=-1.0, gamma=0.0, lambdah=0.5),
        ]
        clamped = [e.run_config.binary_parameters for e in experiments]

        eta_vals, gamma_vals, lambdah_vals = _clamp_summary(clamped)

        assert eta_vals == [-5.0, -3.0, -1.0], f"Got {eta_vals}"
        assert gamma_vals == [-1.0, -0.5, 0.0], f"Got {gamma_vals}"
        assert lambdah_vals == [0.0, 0.5], f"Got {lambdah_vals}"

    def test_single_config(self):
        """A single clamped config should produce a single value per field."""
        experiments = [_make_mock_experiment(eta=2.0, gamma=3.0, lambdah=4.0)]
        clamped = [e.run_config.binary_parameters for e in experiments]

        eta_vals, gamma_vals, lambdah_vals = _clamp_summary(clamped)

        assert eta_vals == [2.0]
        assert gamma_vals == [3.0]
        assert lambdah_vals == [4.0]

    def test_empty_input(self):
        """An empty list of clamped configs should produce empty results."""
        eta_vals, gamma_vals, lambdah_vals = _clamp_summary([])

        assert eta_vals == []
        assert gamma_vals == []
        assert lambdah_vals == []

    def test_without_item_leads_to_identity_based_dedup(self):
        """Regression check: without ``.item()``, a ``set`` of 0-d tensors
        uses identity-based hashing (at least on PyTorch ≥ 2.0). Two tensors
        with the same float value will be distinct set members, causing the
        summary to report duplicate values.

        This test verifies that property so we have a canary if PyTorch ever
        changes tensor hashing semantics — it asserts the *current* identity
        behaviour, not the desired behaviour.
        """
        c1 = _make_mock_experiment(eta=1.0, gamma=2.0, lambdah=3.0).run_config.binary_parameters
        c2 = _make_mock_experiment(eta=1.0, gamma=2.0, lambdah=3.0).run_config.binary_parameters

        # Side-by-side: without .item() vs with .item()
        without_item = sorted(set(c.eta for c in [c1, c2]))
        with_item = sorted(set(c.eta.item() for c in [c1, c2]))

        # With .item() — values are equal floats → deduped
        assert with_item == [1.0], f"BUG: .item() dedup failed: {with_item}"
        # Without .item() — tensors hash by identity → two entries
        assert len(without_item) == 2, (
            f"PyTorch behaviour changed: set() of tensors now deduplicated "
            f"({len(without_item)} entries for 2 tensors with the same value). "
            f"Remove .item() calls if PyTorch now value-hashes tensors."
        )
