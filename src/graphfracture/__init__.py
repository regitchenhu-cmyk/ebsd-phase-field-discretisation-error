"""DOLFINx research backend for graph-informed phase-field fracture.

The pre-existing :mod:`inverse_pfm` package remains the low-fidelity pixel
prototype.  This package provides the independent finite-element baseline and
keeps three graph roles explicit:

``G_grain``
    Grain adjacency and attributes from EBSD.
``G_GB``
    Grain-boundary junction/segment graph used to construct material fields.
``G_crack``
    Active finite-element graph extracted from the converged damage field.
"""

from .chain_field import ChainBoundaryField
from .config import RunConfig, load_config
from .gb_graph import (
    BoundaryEdge,
    BoundaryField,
    BoundaryGraph,
    BoundaryNode,
    boundary_field_from_config,
)

__all__ = [
    "BoundaryEdge",
    "BoundaryField",
    "BoundaryGraph",
    "BoundaryNode",
    "ChainBoundaryField",
    "RunConfig",
    "boundary_field_from_config",
    "load_config",
]

__version__ = "0.2.0"
