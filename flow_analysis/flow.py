#!/usr/bin/env python3

import hashlib
from os.path import basename

from collections import namedtuple

from numpy import asarray
from numpy.random import default_rng

class FlowEnsemble:
    """
    Represents the data from the gradient flow for a single ensemble.
    """

    _frozen = False

    def __init__(self, filename):
        self.trajectories = []
        self.Eps = []
        self.Ecs = []
        self.times = None
        self.Qs = []
        self.metadata = {}
        self.filename = filename

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
            raise ValueError("Times must match for all flows.")

        if not (len(flow.times) == len(flow.Eps) == len(flow.Ecs) == len(flow.Qs)):
            raise ValueError("Flow not a consistent length.")

        self.trajectories.append(flow.trajectory)
        self.Eps.append(flow.Eps)
        self.Ecs.append(flow.Ecs)
        self.Qs.append(flow.Qs)

    def freeze(self):
        """
        Turn the lists of data into Numpy arrays for faster operations.
        """
        self.trajectories = asarray(self.trajectories)
        self.times = asarray(self.times)
        self.Eps = asarray(self.Eps)
        self.Ecs = asarray(self.Ecs)
        self.Qs = asarray(self.Qs)

        self._frozen = True

    def Q_history(self, t="L/2"):
        """
        Get the topological charge Q for each configuration in the ensemble.

        Arguments:

            t: The flow time at which Q is measured.
               If "L/2" is passed (the default), then the relation
               \sqrt{8t} ≤ L / 2 is used to determine t.
        """
        if t == "L/2":
            L = min(self.metadata["NX"], self.metadata["NY"], self.metadata["NZ"])
            t = L ** 2 / 32

        t_index = (self.times <= t).nonzero()[0][-1]
        return self.Qs[:, t_index]


class Flow:
    """
    Represents the data from the gradient flow for a single configuration.
    """

    def __init__(self, trajectory=None):
        self.trajectory = trajectory
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
