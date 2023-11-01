#!/usr/bin/env python3

from functools import lru_cache
from re import findall, match

from numpy import nan

from ..flow import FlowStep, Flow, FlowEnsemble


def add_metadata(metadata, line_contents):
    if line_contents[7:10] == ["Global", "lattice", "size"]:
        metadata["NX"] = int(line_contents[11])
        metadata["NY"] = int(line_contents[12])
        metadata["NZ"] = int(line_contents[13])
        metadata["NT"] = int(line_contents[14])


@lru_cache(maxsize=8)
def read_flows_grid(filename):
    flows = FlowEnsemble(filename)
    flow = None
    Ep_idx = None
    Ec_idx = None
    Q_idx = None

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()

            if len(line_contents) < 8:
                continue

            if len(line_contents) > 8 and line_contents[8] == "Configuration" and line_contents[-1] == "agree":
                if flow:
                    flows.append(flow)

                trajectory = int(findall(".*\.(\d+)$", line_contents[9])[0])
                flow = Flow(trajectory)

            add_metadata(flows.metadata, line_contents)

            if line_contents[7] != "[WilsonFlow]":
                continue

            if line_contents[8:11] == ["Energy", "density", "(plaq)"]:
                Ep_idx = int(line_contents[12])
                flow_time = float(line_contents[13])
                Ep = float(line_contents[14]) / flow_time ** 2
            elif line_contents[8:11] == ["Energy", "density", "(cloverleaf)"]:
                Ec_idx = int(line_contents[12])
                flow_time = float(line_contents[13])
                Ec = float(line_contents[14]) / flow_time ** 2
            elif line_contents[8:10] == ["Top.", "charge"]:
                Q_idx = int(line_contents[11])
                Q = float(line_contents[12])

            if (Ep_idx is not None or Ec_idx is not None) and Ep_idx == Q_idx:
                flow.append(FlowStep(flow_time, Ep or nan, Ec or nan, Q))
                Ep_idx = None
                Ec_idx = None
                Q_idx = None

    flows.append(flow)
    flows.freeze()
    return flows
