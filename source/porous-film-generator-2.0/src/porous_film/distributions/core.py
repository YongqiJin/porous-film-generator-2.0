from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy import stats


def allocate_largest_remainder(weights: Any, total: int) -> np.ndarray:
    if total < 0:
        raise ValueError("total must be nonnegative")

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if np.any(weights_array < 0):
        raise ValueError("weights must be nonnegative")
    weight_sum = float(weights_array.sum())
    if weight_sum <= 0:
        raise ValueError("weights must include positive mass")

    exact = weights_array / weight_sum * total
    counts = np.floor(exact).astype(int)
    remainder_count = total - int(counts.sum())
    if remainder_count:
        order = np.lexsort((np.arange(weights_array.size), -(exact - counts)))
        counts[order[:remainder_count]] += 1
    return counts


def stratified_sample(spec: Any, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return np.array([], dtype=float)

    spec_dict = _as_dict(spec)
    if spec_dict["family"] == "mixture":
        components = spec_dict.get("components") or ()
        weights = np.array([_as_dict(component)["weight"] for component in components], dtype=float)
        component_counts = allocate_largest_remainder(weights, count)
        samples = [
            stratified_sample(_component_distribution(component), int(component_count), rng)
            for component, component_count in zip(components, component_counts, strict=True)
            if component_count
        ]
        if not samples:
            return np.array([], dtype=float)
        combined = np.concatenate(samples)
        return combined[rng.permutation(combined.size)]

    quantiles = (np.arange(count, dtype=float) + rng.random(count)) / count
    return _ppf(spec_dict, quantiles)


def mixture_cdf(spec: Any, values: Any) -> np.ndarray:
    spec_dict = _as_dict(spec)
    values_array = np.asarray(values, dtype=float)
    if spec_dict["family"] == "mixture":
        components = _validated_mixture_components(spec_dict)
        result = np.zeros_like(values_array, dtype=float)
        for component in components:
            component_dict = _as_dict(component)
            result += component_dict["weight"] * mixture_cdf(
                _component_distribution(component_dict),
                values_array,
            )
        return result
    return _cdf(spec_dict, values_array)


def _as_dict(spec: Any) -> dict[str, Any]:
    if isinstance(spec, BaseModel):
        return spec.model_dump(exclude_none=True)
    if isinstance(spec, dict):
        return {key: value for key, value in spec.items() if value is not None}
    raise TypeError("distribution spec must be a mapping or Pydantic model")


def _component_distribution(component: Any) -> dict[str, Any]:
    component_dict = _as_dict(component)
    return {key: value for key, value in component_dict.items() if key != "weight"}


def _validated_mixture_components(spec: dict[str, Any]) -> tuple[Any, ...]:
    components = tuple(spec.get("components") or ())
    if not components:
        raise ValueError("mixture components must be non-empty")

    weights = []
    for component in components:
        component_dict = _as_dict(component)
        if "weight" not in component_dict:
            raise ValueError("mixture weights are required")
        weights.append(float(component_dict["weight"]))

    weights_array = np.asarray(weights, dtype=float)
    if (
        not np.all(np.isfinite(weights_array))
        or np.any(weights_array < 0)
        or abs(float(weights_array.sum()) - 1.0) > 1e-9
    ):
        raise ValueError("mixture weights must be nonnegative and sum to 1")
    return components


def _ppf(spec: dict[str, Any], quantiles: np.ndarray) -> np.ndarray:
    family = spec["family"]
    if family == "constant":
        return np.full_like(quantiles, fill_value=float(spec["value"]), dtype=float)
    return _scipy_distribution(spec).ppf(quantiles)


def _cdf(spec: dict[str, Any], values: np.ndarray) -> np.ndarray:
    family = spec["family"]
    if family == "constant":
        return (values >= float(spec["value"])).astype(float)
    return _scipy_distribution(spec).cdf(values)


def _scipy_distribution(spec: dict[str, Any]) -> Any:
    family = spec["family"]
    if family == "lognormal":
        sigma = _required_float(spec, "sigma", "s")
        loc = float(spec.get("loc", 0.0))
        if "scale" in spec:
            scale = float(spec["scale"])
        else:
            scale = float(np.exp(float(spec.get("mean", spec.get("mu", 0.0)))))
        return stats.lognorm(s=sigma, loc=loc, scale=scale)
    if family == "gamma":
        shape = _required_float(spec, "alpha", "shape", "k")
        scale = float(spec.get("scale", spec.get("theta", 1.0)))
        loc = float(spec.get("loc", 0.0))
        return stats.gamma(a=shape, loc=loc, scale=scale)
    if family in {"weibull", "weibull_min"}:
        shape = _required_float(spec, "shape", "k", "alpha")
        scale = float(spec.get("scale", 1.0))
        loc = float(spec.get("loc", 0.0))
        return stats.weibull_min(c=shape, loc=loc, scale=scale)
    if family in {"truncated_normal", "truncnorm"}:
        mean = float(spec.get("mean", spec.get("loc", 0.0)))
        sigma = _required_float(spec, "sigma", "s")
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        return stats.truncnorm(
            a=(lower - mean) / sigma,
            b=(upper - mean) / sigma,
            loc=mean,
            scale=sigma,
        )
    if family == "beta":
        alpha = _required_float(spec, "alpha")
        beta = _required_float(spec, "beta")
        lower = float(spec.get("lower", spec.get("minimum", 0.0)))
        if "upper" in spec:
            upper = float(spec["upper"])
            scale = upper - lower
        elif "maximum" in spec:
            upper = float(spec["maximum"])
            scale = upper - lower
        else:
            scale = float(spec.get("scale", 1.0))
        return stats.beta(a=alpha, b=beta, loc=lower, scale=scale)
    raise ValueError(f"unsupported distribution family: {family}")


def _required_float(spec: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in spec:
            return float(spec[name])
    joined_names = ", ".join(names)
    raise ValueError(f"distribution requires one of: {joined_names}")
