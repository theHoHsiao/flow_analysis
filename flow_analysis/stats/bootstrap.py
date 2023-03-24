#!/usr/bin/env python3

from numpy import ndarray, mean, std, arccosh, asarray, empty
from numpy.random import default_rng

from uncertainties import ufloat

BOOTSTRAP_SAMPLE_COUNT = 200

# Note: Using default RNG will not give exactly reproducible output
DEFAULT_RNG = default_rng()


def bootstrap_finalize(samples):
    return ufloat(mean(samples), std(samples))


def basic_bootstrap(values, rng=DEFAULT_RNG):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        samples.append(mean(rng.choice(values, len(values))))
    return bootstrap_finalize(samples)


def bootstrap_susceptibility(values, rng=DEFAULT_RNG):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        current_sample = rng.choice(values, len(values))
        samples.append(mean(current_sample**2) - mean(current_sample) ** 2)
    return bootstrap_finalize(samples)


def sample_bootstrap_1d(values, rng=DEFAULT_RNG):
    values = asarray(values)
    bootstrap_sample_configurations = rng.integers(
        values.shape[0], size=(BOOTSTRAP_SAMPLE_COUNT, values.shape[0])
    )
    bootstrap_samples = empty((BOOTSTRAP_SAMPLE_COUNT, values.shape[1]))
    for t_index in range(values.shape[1]):
        bootstrap_samples[:, t_index] = values[
            bootstrap_sample_configurations, t_index
        ].mean(axis=1)
    return asarray(bootstrap_samples)
