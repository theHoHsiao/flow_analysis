#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

from ..fit_forms import exp_decay


def autocorr(series, cutoff=None):
    """
    Calculate the autocorrelation function of the `series`.
    If `cutoff` isn't specified, it defaults to `len(series) // 2`.
    """

    if not cutoff:
        cutoff = len(series) // 2
    acf = np.zeros(cutoff - 1)
    zero_centered_series = series - np.mean(series)
    acf[0] = 1
    for i in range(1, cutoff - 1):
        acf[i] = np.mean(zero_centered_series[:-i] * zero_centered_series[i:]) / np.var(
            zero_centered_series
        )
    return acf


def exp_autocorrelation_fit(series, fit_range=10):
    """
    Fits the exponential autocorrelation time of the first `fit_range` points
    of `series` and returns it with its uncertainty.
    """

    acf = autocorr(series)
    fit_result = curve_fit(exp_decay, np.arange(fit_range), acf[:fit_range])
    tau_exp, tau_exp_error = fit_result[0][0], fit_result[1][0][0] ** 0.5

    if tau_exp < 1 and tau_exp_error > 10 * tau_exp:
        if np.std(acf[1:fit_range]) > np.mean(acf[1:fit_range]):
            # Autocorrelation time is much less than 1
            # Not enough resolution for fitter to determine precise location
            tau_exp = 0
            tau_exp_error = 0.5

    return ufloat(tau_exp, tau_exp_error)
