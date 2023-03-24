#!/usr/bin/env python3

from functools import lru_cache
from re import findall, match

from ..flow import FlowStep, Flow, FlowEnsemble


def add_metadata(metadata, line_contents):
    if (
        line_contents[0] == "[GEOMETRY][0]Global"
        or line_contents[0] == "[GEOMETRY_INIT][0]Global"
    ):
        NT, NX, NY, NZ = map(
            int, match("([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)", line_contents[3]).groups()
        )
        metadata["NT"] = NT
        metadata["NX"] = NX
        metadata["NY"] = NY
        metadata["NZ"] = NZ


@lru_cache(maxsize=8)
def read_flows_hirep(filename):
    flows = FlowEnsemble(filename)
    flow = None

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()

            if (
                line_contents[0] == "[IO][0]Configuration"
                and line_contents[2] == "read"
            ):
                if flow:
                    flows.append(flow)

                trajectory = int(findall(r".*n(\d+)]", line_contents[1])[0])
                flow = Flow(trajectory)

            add_metadata(flows.metadata, line_contents)

            if line_contents[0] != "[WILSONFLOW][0]WF":
                continue

            if line_contents[1].startswith("(ncnfg"):
                # There are two versions of HiRep flow logs
                # One has an extra field that can safely be ignored
                del line_contents[3]

            flow_time = float(line_contents[3])

            Ep = float(line_contents[4])
            Ec = float(line_contents[6])
            Q = float(line_contents[8])

            flow.append(FlowStep(flow_time, Ep, Ec, Q))

    flows.append(flow)
    flows.freeze()
    return flows
