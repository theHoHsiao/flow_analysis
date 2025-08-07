#!/usr/bin/env python3

from collections import Counter

import numpy as np
from scipy.optimize import curve_fit

from uncertainties import ufloat

from ..flow import FlowEnsemble
from ..fit_forms import gaussian
from ..stats.autocorrelation import exp_autocorrelation_fit
from ..stats.bootstrap import basic_bootstrap, bootstrap_susceptibility


def Q_mean(flow_ensemble, t="L/2"):
    """
    Compute the mean and bootstrap error Q of an ensemble.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
        t: The flow time at which Q is measured.
           If "L/2" is passed (the default), then the relation
           \\sqrt{8t} ≤ L / 2 is used to determine t.
    """

    return basic_bootstrap(flow_ensemble.Q_history(t), rng=flow_ensemble.get_rng())


def Q_susceptibility(flow_ensemble, t="L/2"):
    """
    Compute the mean and bootstrap error of the naive
    topological susceptibility of an ensemble.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
        t: The flow time at which Q is measured.
           If "L/2" is passed (the default), then the relation
           \\sqrt{8t} ≤ L / 2 is used to determine t.
    """

    metadata = flow_ensemble.metadata
    V = metadata["NX"] * metadata["NY"] * metadata["NZ"] * metadata["NT"]
    unnorm_suscept = bootstrap_susceptibility(
        flow_ensemble.Q_history(t), rng=flow_ensemble.get_rng()
    )
    return unnorm_suscept / V


def flat_bin_Qs(Q_history, t=None):
    """
    Given a Monte Carlo time history of Q values,
    bin them to integers.

    Arguments:

        Q_history: A list/1D array of Q values, or a FlowEnsemble.
        t: The flow time at which Q is measured.
           If "L/2" is passed (the default), then the relation
           \\sqrt{8t} ≤ L / 2 is used to determine t.
           Only allowed if Q_history is a FlowEnsemble.

    Returns:

        Q_range: a list of values of Q
        Q_counts: a list of counts of each value of Q
    """

    if isinstance(Q_history, FlowEnsemble):
        if t is None:
            t = "L/2"
        Q_history = Q_history.Q_history(t)
    elif t is not None:
        raise ValueError("Cannot recompute Q at a different flow time.")

    Q_bins = Counter(Q_history.round())

    range_min = min(min(Q_bins), -max(Q_bins)) - 1
    range_max = -range_min + 1
    Q_range = np.arange(range_min, range_max)

    # Turn sparse Counter object into dense array
    Q_counts = np.asarray([Q_bins[Q] for Q in Q_range])

    return Q_range, Q_counts


def Q_fit(flow_ensemble, t="L/2", with_amplitude=False):
    """
    Fit a Gaussian to the topological charge distribution of an ensemble,
    and return its estimated centre and width.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
        t: The flow time at which Q is measured.
           If "L/2" is passed (the default), then the relation
           \\sqrt{8t} ≤ L / 2 is used to determine t.
    """

    Q_history = flow_ensemble.Q_history(t)

    Q_range, Q_counts = flat_bin_Qs(Q_history)

    # Estimate a sensible starting point for the fit
    # so it doesn't give up if the peak is a long way away from the default
    Q_mean = Q_history.mean()
    Q_std = Q_history.std()
    Q_amplitude = Q_counts.max()

    popt, pcov = curve_fit(
        gaussian,
        Q_range,
        Q_counts,
        sigma=(Q_counts + 1) ** 0.5,
        p0=[Q_amplitude, Q_mean, Q_std],
        absolute_sigma=True,
    )

    A, Q0, sigma = map(ufloat, popt, pcov.diagonal() ** 0.5)

    if with_amplitude:
        return A, Q0, sigma
    else:
        return Q0, sigma


def main():
    from argparse import ArgumentParser
    from ..readers import readers

    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--reader", default=readers["hirep"], type=readers.get)
    args = parser.parse_args()

    flows = args.reader(args.filename)

    print(f"Q: {Q_mean(flows):.02uSL}")
    print(f"χ: {Q_susceptibility(flows):.02uSL}")

    Q0, sigma = Q_fit(flows)
    print(f"Q0: {Q0:.02uSL}; σ: {sigma:.02uSL}")

    Q_tau_exp = exp_autocorrelation_fit(flows.Q_history())
    print(f"tau_exp: {Q_tau_exp:.02uSL}")


if __name__ == "__main__":
    main()
