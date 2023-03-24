#!/usr/bin/env python3

from collections import Counter

import numpy as np
from scipy.optimize import curve_fit

from uncertainties import ufloat

from ..fit_forms import gaussian
from ..stats.autocorrelation import exp_autocorrelation_fit
from ..stats.bootstrap import basic_bootstrap, bootstrap_susceptibility


def Q_mean(flow_ensemble):
    """
    Compute the mean and bootstrap error Q of an ensemble.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
    """

    return basic_bootstrap(flow_ensemble.Q_history(), rng=flow_ensemble.get_rng())


def chi_top(flow_ensemble):
    """
    Compute the mean and bootstrap error of the naive
    topological susceptibility of an ensemble.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
    """

    metadata = flow_ensemble.metadata
    V = metadata["NX"] * metadata["NY"] * metadata["NZ"] * metadata["NT"]
    unnorm_suscept = bootstrap_susceptibility(
        flow_ensemble.Q_history(), rng=flow_ensemble.get_rng()
    )
    return unnorm_suscept / V


def flat_bin_Qs(Q_history):
    """
    Given a Monte Carlo time history of Q values,
    bin them to integers.

    Arguments:

        Q_history: A list/1D array of Q values.

    Returns:

        Q_range: a list of values of Q
        Q_counts: a list of counts of each value of Q
    """

    Q_bins = Counter(Q_history.round())

    range_min = min(min(Q_bins), -max(Q_bins)) - 1
    range_max = -range_min + 1
    Q_range = np.arange(range_min, range_max)

    # Turn sparse Counter object into dense list
    Q_counts = [Q_bins[Q] for Q in Q_range]

    return Q_range, Q_counts


def Q_fit(flow_ensemble):
    """
    Fit a Gaussian to the topological charge distribution of an ensemble,
    and return its estimated centre and width.

    Arguments:

        flow_ensemble: A frozen FlowEnsemble instance.
    """

    Q_range, Q_counts = flat_bin_Qs(flow_ensemble.Q_history())
    popt, pcov = curve_fit(gaussian, Q_range, Q_counts)

    A, Q0, sigma = map(ufloat, popt, pcov.diagonal() ** 0.5)

    return Q0, sigma


def main():
    from argparse import ArgumentParser
    from ..readers import read_flows_hirep

    parser = ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    flows = read_flows_hirep(args.filename)

    print(f"Q: {Q_mean(flows):.02uSL}")
    print(f"χ: {chi_top(flows):.02uSL}")

    Q0, sigma = Q_fit(flows)
    print(f"Q0: {Q0:.02uSL}; σ: {sigma:.02uSL}")

    Q_tau_exp = exp_autocorrelation_fit(flows.Q_history())
    print(f"tau_exp: {Q_tau_exp:.02uSL}")


if __name__ == "__main__":
    main()
