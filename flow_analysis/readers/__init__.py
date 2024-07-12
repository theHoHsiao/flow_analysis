#!/usr/bin/env python3

from .read_grid import read_flows_grid
from .read_hirep import read_flows_hirep
from .read_hp import read_flows_hp


readers = {
    "hirep": read_flows_hirep,
    "grid": read_flows_grid,
    "hp": read_flows_hp,
}
