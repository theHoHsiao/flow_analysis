#!/usr/bin/env python3

import hashlib
from os.path import basename

from collections import Counter, namedtuple

from numpy import asarray
from numpy.random import default_rng

import pyerrors as pe


class FlowEnsemble:
    """
    Represents the data from the gradient flow for a single ensemble.
    """

    _frozen = False

    def __init__(self, filename, reader=None):
        self.ensemble_names = []
        self.trajectories = []
        self.reader = reader
        self.Eps = []
        self.Ecs = []
        self.times = None
        self.Qs = []
        self.metadata = {}
        self.filename = filename

    def __len__(self):
        return len(self.trajectories)

    def get_rng(self):
        """
        Use the base filename of the input file to generate
        a consistent for generating random numbers.
        """

        filename = basename(self.filename)
        filename_hash = hashlib.md5(filename.encode("utf8")).digest()
        seed = abs(int.from_bytes(filename_hash, "big"))
        return default_rng(seed)

    def append(self, flow):
        """
        Append the flow for one configuration to the data for the ensemble.

        Arguments:
            flow: The instance of Flow to append.
        """
        if self._frozen:
            raise TypeError("Can't append to a frozen ensemble.")

        if self.times is None:
            self.times = flow.times
        elif flow.times != self.times:
            raise ValueError(
                f"Times must match for all flows. (Failing at trajectory {flow.trajectory})"
            )

        if not (len(flow.times) == len(flow.Eps) == len(flow.Ecs) == len(flow.Qs)):
            raise ValueError(
                f"Flow for trajectory {flow.trajectory} not a consistent length."
            )

        self.ensemble_names.append(flow.ensemble)
        self.trajectories.append(flow.trajectory)
        self.Eps.append(flow.Eps)
        self.Ecs.append(flow.Ecs)
        self.Qs.append(flow.Qs)

    def freeze(self):
        """
        Turn the lists of data into Numpy arrays for faster operations.
        """
        self.ensemble_names = asarray(self.ensemble_names)
        self.trajectories = asarray(self.trajectories)
        self.times = asarray(self.times)
        self.Eps = asarray(self.Eps)
        self.Ecs = asarray(self.Ecs)
        self.Qs = asarray(self.Qs)

        self._frozen = True

    def thin(self, min_trajectory=None, max_trajectory=None, trajectory_step=1):
        """
        Thin an ensemble to decorrelate it.
        """
        if not self._frozen:
            raise NotImplementedError(
                "Thinning an unfrozen ensemble isn't currently supported"
            )

        mask = (
            (self.trajectories > min_trajectory if min_trajectory is not None else True)
            & (
                self.trajectories < max_trajectory
                if max_trajectory is not None
                else True
            )
            & (
                (self.trajectories - (min_trajectory if min_trajectory else 0))
                % trajectory_step
                == 0
            )
        )
        result = FlowEnsemble(self.filename, self.reader)
        result.ensemble_names = self.ensemble_names[mask]
        result.trajectories = self.trajectories[mask]
        result.times = self.times
        result.Eps = self.Eps[mask, :]
        result.Ecs = self.Ecs[mask, :]
        result.Qs = self.Qs[mask, :]
        result._frozen = True
        result.metadata = self.metadata

        return result

    def group(self, observable):
        return [
            observable[self.ensemble_names == ensemble_name]
            for ensemble_name in sorted(set(self.ensemble_names))
        ]

    def Q_history(self, t="L/2"):
        """
        Get the topological charge Q for each configuration in the ensemble.

        Arguments:

            t: The flow time at which Q is measured.
               If "L/2" is passed (the default), then the relation
               \\sqrt{8t} ≤ L / 2 is used to determine t.
        """
        if t == "L/2":
            L = min(self.metadata["NX"], self.metadata["NY"], self.metadata["NZ"])
            t = L**2 / 32

        t_index = (self.times <= t).nonzero()[0][-1]
        return self.Qs[:, t_index]

    @property
    def is_adaptive(self):
        tolerance = 1e-5
        return (self.hs.max() - self.hs.min()) / self.hs.mean() > tolerance

    @property
    def hs(self):
        times = asarray(self.times)
        return times[1:] - times[:-1]

    @property
    def h(self):
        if self.is_adaptive:
            raise ValueError("Can't get single step size of adaptive flow.")
        return self.hs.mean()

    def get_Es(self, operator):
        """
        Get the values of the energy density for a given operator.

        Arguments:
            operator: The operator for E to use.
                      Valid options are "plaq" and "sym".
        """
        if operator == "plaq":
            return self.Eps
        elif operator == "sym":
            return self.Ecs
        else:
            raise ValueError(
                f"Invalid operator {operator}. " 'Valid operators are "plaq" and "sym".'
            )

    def get_Es_pyerrors(self, operator, tag=None):
        """
        Get a pyerrors object encapsulating the ensemble of energy density computations
        for a a given operator.

        Arguments:
            operator: The operator for E to use.
                      Valid options are "plaq" and "sym".
        """

        Es = self.get_Es(operator)
        all_samples = [[] for _ in range(len(self.times))]
        ensemble_names = sorted(set(self.ensemble_names))
        sample_idxs = []

        for ensemble_name in ensemble_names:
            subset_idx = self.ensemble_names == ensemble_name
            ensemble_subset = Es[subset_idx]
            trajectories_subset = self.trajectories[subset_idx]
            if len(set(trajectories_subset)) != len(trajectories_subset):
                counts = Counter(trajectories_subset)
                duplicate_trajectories = [
                    index for index, count in counts.items() if count > 1
                ]
                message = f"Can't make a pyerrors object for the ensemble {self.filename} as there are duplicate trajectories: {duplicate_trajectories}"
                raise ValueError(message)

            for samples in all_samples:
                samples.append([])

            sample_idxs.append([])
            for trajectory, flow in zip(trajectories_subset, ensemble_subset):
                sample_idxs[-1].append(trajectory)
                for samples, E_value in zip(all_samples, flow):
                    samples[-1].append(E_value)

        observables = [
            pe.Obs(samples, ensemble_names, idl=sample_idxs) for samples in all_samples
        ]

        E_flows_pe = pe.Corr(observables)
        E_flows_pe.tag = tag

        return E_flows_pe


class Flow:
    """
    Represents the data from the gradient flow for a single configuration.
    """

    def __init__(self, trajectory=None, ensemble=""):
        self.trajectory = trajectory
        self.ensemble = ensemble
        self.Eps = []
        self.Ecs = []
        self.times = []
        self.Qs = []

    def append(self, flowstep):
        """
        Append the data for one flow time step to the that for the configuration.

        Arguments:
            flowstep: The instance of FlowStep to append.
        """

        if self.times and flowstep.t < self.times[-1]:
            raise ValueError("Flow goes backwards.")
        else:
            self.times.append(flowstep.t)

        self.Eps.append(flowstep.Ep)
        self.Ecs.append(flowstep.Ec)
        self.Qs.append(flowstep.Q)


FlowStep = namedtuple("FlowStep", ["t", "Ep", "Ec", "Q"])
