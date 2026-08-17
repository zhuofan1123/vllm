# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RadixShmem-backed shared CPU KV cache for single-node DP x TP."""

from .bootstrap import SharedRegions, attach_regions, create_regions, sentinel_path
from .geometry import (
    GeometryMismatch,
    SlotGeometry,
    TensorLayout,
    compute_geometry,
    tensor_layout,
    verify_against_canonical,
)

__all__ = [
    "GeometryMismatch",
    "SharedRegions",
    "SlotGeometry",
    "TensorLayout",
    "attach_regions",
    "compute_geometry",
    "create_regions",
    "sentinel_path",
    "tensor_layout",
    "verify_against_canonical",
]
