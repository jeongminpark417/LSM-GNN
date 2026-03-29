#!/usr/bin/env python3
"""
PyG ``NeighborLoader`` helpers that avoid collating full-dim node features.

``NeighborLoader`` always expects a ``Data.x`` tensor; for NVMe/GIDS training you
only need ``batch.n_id`` plus graph structure. This module builds a shallow copy
of ``Data`` **without copying** the original ``x`` storage: it sets
``x = zeros(num_nodes, 0)`` so sampling moves **O(subgraph × 0)** feature bytes
instead of ``O(subgraph × feat_dim)``. Real embeddings must come from
``gids.fetch_feature(batch.n_id, feat_dim)`` (or similar).

Example::

    from torch_geometric.loader import NeighborLoader
    from pyg_loader_skip_features import data_without_dense_node_features, NeighborLoaderSkipFeatures

    slim = data_without_dense_node_features(data)
    loader = NeighborLoaderSkipFeatures(
        slim,
        num_neighbors=[10, 10],
        batch_size=1024,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=0,
    )

    # Same as NeighborLoader(slim, ...) — subclass is optional sugar.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch_geometric.data import Data

try:
    from torch_geometric.loader import NeighborLoader
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for pyg_loader_skip_features"
    ) from e


def data_without_dense_node_features(
    data: Data,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Data:
    """
    New ``Data`` with the same non-``x`` tensors as ``data``, and
    ``x`` of shape ``(num_nodes, 0)`` (no dense features copied).

    The original ``data.x`` tensor is **not** referenced by the returned object.
    """
    n = int(data.num_nodes)
    dev = device if device is not None else data.edge_index.device
    out = Data()
    for key in data.keys():
        if key == "x":
            continue
        out[key] = data[key]
    out.x = torch.zeros((n, 0), dtype=dtype, device=dev)
    return out


class NeighborLoaderSkipFeatures(NeighborLoader):
    """
    ``NeighborLoader`` on a graph with **zero-width** ``x`` (see
    :func:`data_without_dense_node_features`).

    Pass your full ``Data`` and set ``skip_features=True`` (default) to strip
    dense features before constructing the underlying loader. Pass ``False`` if
    ``data`` is already slim.
    """

    def __init__(
        self,
        data: Data,
        *args: Any,
        skip_features: bool = True,
        feature_dtype: torch.dtype = torch.float32,
        feature_device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        if skip_features:
            data = data_without_dense_node_features(
                data, dtype=feature_dtype, device=feature_device
            )
        super().__init__(data, *args, **kwargs)


def neighbor_loaders_skip_features(
    data: Data,
    *,
    num_neighbors: Iterable[int],
    batch_size: int,
    num_workers: int = 0,
    neighbor_loader_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[NeighborLoaderSkipFeatures, NeighborLoaderSkipFeatures, NeighborLoaderSkipFeatures]:
    """
    Train / val / test ``NeighborLoaderSkipFeatures`` using ``train_mask``,
    ``val_mask``, and ``test_mask`` on ``data``.

    Slims ``data`` once (``data_without_dense_node_features``) and reuses it for
    all three loaders so large ``x`` is never copied.
    """
    slim = data_without_dense_node_features(data)
    kw: dict[str, Any] = dict(
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if neighbor_loader_kwargs:
        kw.update(neighbor_loader_kwargs)
    train = NeighborLoaderSkipFeatures(
        slim, input_nodes=data.train_mask, shuffle=True, skip_features=False, **kw
    )
    val = NeighborLoaderSkipFeatures(
        slim, input_nodes=data.val_mask, shuffle=False, skip_features=False, **kw
    )
    test = NeighborLoaderSkipFeatures(
        slim, input_nodes=data.test_mask, shuffle=False, skip_features=False, **kw
    )
    return train, val, test
