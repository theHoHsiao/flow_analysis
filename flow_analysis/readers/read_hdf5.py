#!/usr/bin/env python3

import h5py

from ..flow import FlowEnsemble


def get_metadata(group):
    NT, NX, NY, NZ = group["lattice"]
    beta = group["beta"][()]
    mAS = group["quarkmasses"][()]
    return {"NT": NT, "NX": NX, "NY": NY, "NZ": NZ, "beta": beta, "mAS": mAS}


def read_flows_hdf5(filename, group_name="/"):
    h5file = h5py.File(filename, "r")
    group = h5file[group_name]

    ensemble = FlowEnsemble(filename, reader="hdf5")
    ensemble.ensemble_names = group["ensemble names"][:]
    ensemble.trajectories = group["trajectory indices"][:]
    ensemble.Eps = group["energy density plaq"][:]
    ensemble.Ecs = group["energy density sym"][:]
    ensemble.Qs = group["topological charge"][:]
    ensemble.times = group["flow times"][:]
    ensemble.cfg_filenames = group["configurations"][:]

    ensemble.metadata = get_metadata(group)
    ensemble.freeze()

    return ensemble
