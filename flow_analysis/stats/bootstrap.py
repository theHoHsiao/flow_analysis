#!/usr/bin/env python3

from numpy import ndarray, mean, std, arccosh, asarray, empty
from numpy.random import default_rng

from uncertainties import ufloat

BOOTSTRAP_SAMPLE_COUNT = 200

# Note: Using default RNG will not give exactly reproducible output
DEFAULT_RNG = default_rng()


def basic_bootstrap(values, rng=DEFAULT_RNG):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        samples.append(mean(rng.choice(values, len(values))))
    return ufloat(mean(samples, axis=0), std(samples))


def bootstrap_susceptibility(values, rng=DEFAULT_RNG):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        current_sample = rng.choice(values, len(values))
        samples.append(mean(current_sample ** 2) - mean(current_sample) ** 2)
    return ufloat(mean(samples, axis=0), std(samples))
