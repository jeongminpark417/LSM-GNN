#!/usr/bin/env python3
"""
Re-exports :class:`LSM_GNN_Neighbor_Loader` from the self-contained
``lsm_gnn_neighbor_loader`` module in this directory (``lsm_nvme=``, optional PVP batch prefetch).

Keeps this import path stable for training scripts; ensure ``lsm_nvme_pyg`` is on
``sys.path`` (this file prepends its directory so the local module wins).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Always prefer this directory first: parent ../pyg also ships ``lsm_gnn_neighbor_loader.py``.
try:
    sys.path.remove(_HERE)
except ValueError:
    pass
sys.path.insert(0, _HERE)

from lsm_gnn_neighbor_loader import LSM_GNN_Neighbor_Loader  # noqa: E402

__all__ = ["LSM_GNN_Neighbor_Loader"]
