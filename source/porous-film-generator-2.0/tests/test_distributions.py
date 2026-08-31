import numpy as np
import pytest

from porous_film.distributions import (
    allocate_largest_remainder,
    mixture_cdf,
    stratified_sample,
)


def test_largest_remainder_preserves_total_and_weights() -> None:
    counts = allocate_largest_remainder(np.array([0.6, 0.3, 0.1]), 17)

    assert counts.tolist() == [10, 5, 2]
    assert int(counts.sum()) == 17


def test_stratified_mixture_is_reproducible_and_covers_both_modes() -> None:
    spec = {
        "family": "mixture",
        "components": [
            {"weight": 0.5, "family": "constant", "value": 0.5},
            {"weight": 0.5, "family": "constant", "value": 2.0},
        ],
    }

    first = stratified_sample(spec, 8, np.random.default_rng(12))
    second = stratified_sample(spec, 8, np.random.default_rng(12))

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [0.5] * 4 + [2.0] * 4


def test_mixture_cdf_combines_constant_components_with_literal_values() -> None:
    spec = {
        "family": "mixture",
        "components": [
            {"weight": 0.25, "family": "constant", "value": 1.0},
            {"weight": 0.75, "family": "constant", "value": 3.0},
        ],
    }

    values = mixture_cdf(spec, np.array([0.0, 1.0, 2.0, 3.0]))

    np.testing.assert_allclose(values, np.array([0.0, 0.25, 0.25, 1.0]))


def test_mixture_cdf_rejects_invalid_direct_call_weights() -> None:
    spec = {
        "family": "mixture",
        "components": [
            {"weight": 1.0, "family": "constant", "value": 1.0},
            {"weight": 0.5, "family": "constant", "value": 3.0},
        ],
    }

    with pytest.raises(ValueError, match="mixture weights"):
        mixture_cdf(spec, np.array([1.0]))


def test_mixture_cdf_rejects_missing_direct_call_components() -> None:
    spec = {"family": "mixture", "components": []}

    with pytest.raises(ValueError, match="mixture components"):
        mixture_cdf(spec, np.array([1.0]))


def test_mixture_cdf_rejects_nonfinite_direct_call_weights() -> None:
    for invalid_weight in (np.nan, np.inf):
        spec = {
            "family": "mixture",
            "components": [
                {"weight": invalid_weight, "family": "constant", "value": 1.0},
                {"weight": 1.0, "family": "constant", "value": 3.0},
            ],
        }

        with pytest.raises(ValueError, match="mixture weights"):
            mixture_cdf(spec, np.array([1.0]))


def test_mixture_cdf_evaluates_beta_uniform_component_with_literal_values() -> None:
    values = mixture_cdf(
        {"family": "beta", "alpha": 1.0, "beta": 1.0},
        np.array([-0.5, 0.0, 0.25, 0.5, 1.0, 1.5]),
    )

    np.testing.assert_allclose(values, np.array([0.0, 0.0, 0.25, 0.5, 1.0, 1.0]))


def test_mixture_cdf_evaluates_nested_constant_mixture_with_literal_values() -> None:
    spec = {
        "family": "mixture",
        "components": [
            {"weight": 0.5, "family": "constant", "value": 1.0},
            {
                "weight": 0.5,
                "family": "mixture",
                "components": [
                    {"weight": 0.25, "family": "constant", "value": 2.0},
                    {"weight": 0.75, "family": "constant", "value": 4.0},
                ],
            },
        ],
    }

    values = mixture_cdf(spec, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    np.testing.assert_allclose(values, np.array([0.0, 0.5, 0.625, 0.625, 1.0]))
