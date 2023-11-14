#!/usr/bin/env python3

from functools import lru_cache
from re import match

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


def parse_cfg_filename(filename):
    """
    Parse out the run name and trajectory index from a configuration filename.

    Arguments:

        filename: The configuration filename

    Returns:

        run_name: The name of the run/stream
        cfg_index: The index of the trajectory in the stream
    """

    matched_filename = match(
        r".*/([^/]*)_[0-9]+x[0-9]+x[0-9]+x[0-9]+nc[0-9]+(?:r[A-Z]+)?(?:nf[0-9]+)?(?:b[0-9]+\.[0-9]+)?(?:m-?[0-9]+\.[0-9]+)?n([0-9]+)",
        filename,
    )
    run_name, cfg_index = matched_filename.groups()
    return run_name, int(cfg_index)


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

                ensemble, trajectory = parse_cfg_filename(line_contents[1])
                flow = Flow(trajectory=trajectory, ensemble=ensemble)

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
