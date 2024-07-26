#!/usr/bin/env python3

from functools import lru_cache

from ..flow import FlowStep, Flow, FlowEnsemble


@lru_cache(maxsize=8)
def read_flows_hp(filename):
    flows = FlowEnsemble(filename, "hp")
    flow = None
    previous_line_trajectory = None
    previous_line_flow_time = None

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()
            trajectory = int(line_contents[0])
            flow_time, Ep, Ec = map(float, line_contents[1:])

            if previous_line_flow_time is None or flow_time < previous_line_flow_time:
                if (
                    previous_line_trajectory
                    and previous_line_trajectory != trajectory - 1
                ):
                    raise ValueError("Configuration indices don't increment nicely.")
                if flow:
                    flows.append(flow)

                flow = Flow(trajectory=trajectory)
                previous_line_flow_time = 0

            flow.append(FlowStep(flow_time, Ep, Ec, None))
            previous_line_trajectory = trajectory
            previous_line_flow_time = flow_time

    flows.append(flow)
    flows.freeze()
    return flows
