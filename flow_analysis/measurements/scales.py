#!/usr/bin/env python3

import warnings

from numpy import argmax

from ..stats.bootstrap import bootstrap_finalize, sample_bootstrap_1d


def _threshold_interpolate(flow_ensemble, values, threshold):
    """
    Find at what time a series of values crosses a given threshold for the first time,
    interpolating where this is between points.

    Arguments:
        flow_ensemble: The FlowEnsemble under consideration..
        values: The values to interpolate, having the same structure as flow_ensemble.Eps.
        threshold: The value to solve for.
    """

    if (threshold <= values[:, 0]).any():
        raise ValueError("Some or all flows start above threshold.")

    positions = argmax(values > threshold, axis=1)

    if min(positions) == 0:
        bad_ratio = sum(positions == 0) / len(positions)
        warnings.warn(f"{bad_ratio:%} of samples do not reach threshold {threshold}")
        positions = positions[positions > 0]

    T_positions_minus_one = values[tuple(zip(*enumerate(positions - 1)))]
    T_positions = values[tuple(zip(*enumerate(positions)))]

    return flow_ensemble.times[positions] + flow_ensemble.h * (
        (threshold - T_positions_minus_one) / (T_positions - T_positions_minus_one)
    )


def bootstrap_ensemble_sqrt_8t0(flow_ensemble, E0, operator="sym"):
    """
    Generate a set of bootstrap samples for an ensemble, and
    compute \sqrt{8t_0} for each sample.

    Arguments:
        flow_ensemble: The FlowEnsemble to evaluate \sqrt{8t_0} for.
        E0: The threshold value E0 to solve for.
        operator: The operator for E to use.
                  Valid options are "plaq" and "sym".
                  Default: sym.
    """

    bs_Es = sample_bootstrap_1d(
        flow_ensemble.get_Es(operator), rng=flow_ensemble.get_rng()
    )
    t2E = flow_ensemble.times**2 * bs_Es
    return (8 * _threshold_interpolate(flow_ensemble, t2E, E0)) ** 0.5


def measure_sqrt_8t0(flow_ensemble, E0, operator="sym"):
    """
    Compute the ensemble average of \sqrt{8t0}.

    Arguments:
        flow_ensemble: The FlowEnsemble to evaluate \sqrt{8t_0} for.
        E0: The threshold value E0 to solve for.
        operator: The operator for E to use.
                  Valid options are "plaq" and "sym".
                  Default: sym.
    """

    return bootstrap_finalize(
        bootstrap_ensemble_sqrt_8t0(flow_ensemble, E0, operator=operator)
    )


def bootstrap_ensemble_w0(flow_ensemble, W0, operator="sym"):
    """
    Generate a set of bootstrap samples for an ensemble, and
    compute w_0 for each sample.

    Arguments:
        flow_ensemble: The FlowEnsemble to evaluate w0 for.
        W0: The threshold value W0 to solve for.
        operator: The operator for E to use.
                  Valid options are "plaq" and "sym".
                  Default: sym.
    """

    bs_Es = sample_bootstrap_1d(
        flow_ensemble.get_Es(operator), rng=flow_ensemble.get_rng()
    )
    times = flow_ensemble.times

    t2E = times**2 * bs_Es
    t_dt2E_dt = times[1:-1] * (t2E[:, 2:] - t2E[:, :-2]) / (2 * flow_ensemble.h)

    return _threshold_interpolate(flow_ensemble, t_dt2E_dt, W0) ** 0.5


def measure_w0(flow_ensemble, W0, operator="sym"):
    """
    Compute the ensemble average of w_0.

    Arguments:
        flow_ensemble: The FlowEnsemble to evaluate w_0 for.
        E0: The threshold value W0 to solve for.
        operator: The operator for E to use.
                  Valid options are "plaq" and "sym".
                  Default: sym.
    """

    return bootstrap_finalize(
        bootstrap_ensemble_w0(flow_ensemble, W0, operator=operator)
    )


def main():
    from argparse import ArgumentParser
    from ..readers import read_flows_hirep

    parser = ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    flows = read_flows_hirep(args.filename)

    threshold = 0.3
    for label, fn in (r"\sqrt{8t_0}", measure_sqrt_8t0), ("w_0", measure_w0):
        for operator in "plaq", "sym":
            observable = fn(flows, threshold, operator=operator)
            print(f"threshold = 0.3, {label} ({operator}) = {observable:.02uSL}")


if __name__ == "__main__":
    main()
