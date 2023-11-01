#!/usr/bin/env python3

from .read_grid import read_flows_grid
from .read_hirep import read_flows_hirep


readers = {
    "hirep": read_flows_hirep,
    "grid": read_flows_grid,
}
