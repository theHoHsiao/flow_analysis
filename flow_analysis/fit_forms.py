#!/usr/bin/env python3

import numpy as np


def gaussian(x, A, x0, sigma):
    """
    Gaussian fit form
    Returns A e^(-(x - x0)^2 / 2 sigma^2)
    """

    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def exp_decay(x, tau):
    """
    The fit form of an exponential decay.
    Returns A e^(-x / tau)
    """

    return np.exp(-x / tau)
