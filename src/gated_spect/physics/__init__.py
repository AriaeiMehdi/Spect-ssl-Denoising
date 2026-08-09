"""Reference SPECT projection physics and iterative reconstruction."""

from .osem import (
    mlem_reconstruct,
    osem_reconstruct,
    poisson_nll,
    subset_view_indices,
)
from .projector import Geometry, IdealParallelProjector

# Public semantic label: this projector is mathematically ideal and is not a
# validated clinical SPECT physics model.
IdealProjector = IdealParallelProjector

__all__ = [
    "Geometry",
    "IdealParallelProjector",
    "IdealProjector",
    "mlem_reconstruct",
    "osem_reconstruct",
    "poisson_nll",
    "subset_view_indices",
]
