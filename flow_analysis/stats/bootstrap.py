#!/usr/bin/env python3

from numpy import ndarray, mean, std, arccosh, asarray, empty
from numpy.random import randint, choice

from uncertainties import ufloat

BOOTSTRAP_SAMPLE_COUNT = 200


def basic_bootstrap(values):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        samples.append(mean(choice(values, len(values))))
    return ufloat(mean(samples, axis=0), std(samples))


def bootstrap_susceptibility(values):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        current_sample = choice(values, len(values))
        samples.append(mean(current_sample ** 2) - mean(current_sample) ** 2)
    return ufloat(mean(samples, axis=0), std(samples))
