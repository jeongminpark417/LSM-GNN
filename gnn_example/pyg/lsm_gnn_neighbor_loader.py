#!/usr/bin/env python3
"""LSM-GNN PyG :class:`~torch_geometric.loader.NeighborLoader` with optional GIDS features.

:class:`LSM_GNN_Neighbor_Loader` overrides :meth:`filter_fn` when ``gids=`` is set
(homogeneous :class:`~torch_geometric.data.Data` only): it uses
:func:`filter_data_without_x` then ``gids.fetch_feature(batch.n_id, ...)``.
Without ``gids``, behavior matches the parent :class:`~torch_geometric.loader.NeighborLoader`.

Optional ``iterator_start_batch`` / ``iterator_max_batches`` limit which seed-node
batches are produced (aligned to ``batch_size``). While iterating, use
``iterator_batch_in_pass`` (0-based within the current ``for`` loop) and
``iterator_logical_batch_index`` (includes the start offset) to see position.

Iteration does **not** use :class:`~torch_geometric.loader.base.DataLoaderIterator`.
Instead, :meth:`__iter__` uses :meth:`torch.utils.data.DataLoader._get_iterator`
while skipping :class:`~torch_geometric.loader.node_loader.NodeLoader` wrapping,
then calls :meth:`filter_fn` only for batches that are actually yielded (skipped
batches in the fallback path never run :meth:`filter_fn`).

See :mod:`pyg_neighbor_prefetch_loader` for a wrapper around an existing loader.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.loader.utils import filter_edge_store_, index_select
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, TensorFrame


def _filter_node_store_skip_x(store, out_store, index: torch.Tensor) -> None:
    """Mirror PyG ``filter_node_store_`` but skip ``x`` (features from GIDS later)."""
    for key, value in store.items():
        if key == "x":
            continue
        if key == "num_nodes":
            out_store.num_nodes = index.numel()
        elif store.is_node_attr(key):
            if isinstance(value, (torch.Tensor, TensorFrame)):
                index = index.to(value.device)
            elif isinstance(value, np.ndarray):
                index = index.cpu()
            dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)


def filter_data_without_x(
    data: Data,
    node: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    edge: OptTensor,
    perm: OptTensor = None,
) -> Data:
    """Subset ``Data`` like PyG ``filter_data`` without touching ``x``."""
    out = copy.copy(data)
    if "x" in out:
        del out["x"]
    _filter_node_store_skip_x(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


class _EmptyLoaderIter:
    """Iterator that yields nothing (matches PyG / DataLoader iter protocol hooks)."""

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __len__(self) -> int:
        return 0

    def _reset(self, loader: Any, first_iter: bool = False) -> None:
        pass


class _RawSamplerIterator:
    """Raw :class:`~torch.utils.data.DataLoader` iterator + explicit :meth:`filter_fn`.

    ``base`` yields collate output (``SamplerOutput`` / ``HeteroSamplerOutput``) before
    :meth:`filter_fn` runs.
    """

    __slots__ = ("_base", "_loader", "_pending_skip", "_yielded", "_limit")

    def __init__(self, base: Any, loader: "LSM_GNN_Neighbor_Loader") -> None:
        self._base = base
        self._loader = loader
        self._pending_skip = int(loader._iterator_fallback_skip_batches)
        self._limit = loader._iterator_fallback_max_batches
        self._yielded = 0
        loader._iterator_batch_in_pass = -1

    def __iter__(self):
        return self

    def _reset(self, loader: Any, first_iter: bool = False) -> None:
        self._pending_skip = int(self._loader._iterator_fallback_skip_batches)
        self._limit = self._loader._iterator_fallback_max_batches
        self._yielded = 0
        self._loader._iterator_batch_in_pass = -1
        if hasattr(self._base, "_reset"):
            self._base._reset(loader, first_iter)

    def __next__(self) -> Any:
        while self._pending_skip > 0:
            try:
                next(self._base)
            except StopIteration:
                self._pending_skip = 0
                raise
            self._pending_skip -= 1
        if self._limit is not None and self._yielded >= self._limit:
            raise StopIteration
        raw = next(self._base)
        self._yielded += 1
        self._loader._iterator_batch_in_pass += 1
        return self._loader.filter_fn(raw)

    def __len__(self) -> int:
        inner_len = len(self._base)
        skip = int(self._loader._iterator_fallback_skip_batches)
        lim = self._loader._iterator_fallback_max_batches
        remain = max(0, inner_len - skip)
        if lim is not None:
            remain = min(remain, lim)
        return remain


class LSM_GNN_Neighbor_Loader(NeighborLoader):
    """Neighbor sampling with optional ``gids`` for batch node features (NVMe / BAM)."""

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        is_sorted: bool = False,
        neighbor_sampler=None,
        directed: bool = True,
        *,
        gids: Any = None,
        gids_feat_dim: Optional[int] = None,
        gids_device: Optional[torch.device] = None,
        gids_timing_stats: Optional[dict[str, Union[float, int]]] = None,
        iterator_start_batch: int = 0,
        iterator_max_batches: Optional[int] = None,
        **kwargs,
    ) -> None:
        if gids is not None:
            if gids_feat_dim is None:
                raise ValueError("gids_feat_dim is required when gids is set")
            if gids_device is None:
                raise ValueError("gids_device is required when gids is set")
            # PyG infers filter_per_worker=True for in-memory CPU graphs. Then
            # NodeLoader.collate_fn runs filter_fn in workers (SamplerOutput→Data).
            # Our __iter__ uses _RawSamplerIterator, which calls filter_fn again on
            # that batch; the second call is not a SamplerOutput and fails. Force
            # filtering in the main process only when gids fills batch.x.
            kwargs["filter_per_worker"] = False

        self._gids = gids
        self._gids_feat_dim = int(gids_feat_dim) if gids_feat_dim is not None else None
        self._gids_device = gids_device
        self._gids_timing_stats = gids_timing_stats

        super().__init__(
            data,
            num_neighbors,
            input_nodes=input_nodes,
            input_time=input_time,
            replace=replace,
            subgraph_type=subgraph_type,
            disjoint=disjoint,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            weight_attr=weight_attr,
            is_sorted=is_sorted,
            neighbor_sampler=neighbor_sampler,
            directed=directed,
            **kwargs,
        )

        self._configure_iterator_limits(iterator_start_batch, iterator_max_batches)

    def _raw_dataloader_iterator(self):
        """``DataLoader`` iterator without :class:`~torch_geometric.loader.base.DataLoaderIterator`."""
        return super(NodeLoader, self)._get_iterator()

    def _configure_iterator_limits(
        self,
        iterator_start_batch: int,
        iterator_max_batches: Optional[int],
    ) -> None:
        self.iterator_start_batch = int(iterator_start_batch)
        self.iterator_max_batches = iterator_max_batches
        self._iterator_batch_in_pass = -1
        self._iterator_empty = False
        self._iterator_fallback_skip_batches = 0
        self._iterator_fallback_max_batches: Optional[int] = None

        if self.iterator_start_batch < 0:
            raise ValueError("iterator_start_batch must be >= 0")
        if iterator_max_batches is not None and iterator_max_batches <= 0:
            raise ValueError("iterator_max_batches must be positive or None")

        if self.iterator_start_batch == 0 and iterator_max_batches is None:
            return

        n_seeds = len(self.dataset)
        bs = self.batch_size
        if bs is None:
            raise ValueError(
                "iterator_start_batch / iterator_max_batches require batch_size"
            )

        start_seed = min(self.iterator_start_batch * bs, n_seeds)
        if iterator_max_batches is None:
            end_seed = n_seeds
        else:
            end_seed = min(start_seed + iterator_max_batches * bs, n_seeds)

        if start_seed >= end_seed:
            self._iterator_empty = True
            return

        can_narrow = isinstance(self.sampler, (SequentialSampler, RandomSampler)) and (
            getattr(self.sampler, "data_source", None) is self.dataset
        )

        if can_narrow:
            new_ds = range(start_seed, end_seed)
            object.__setattr__(self, "dataset", new_ds)
            self.sampler.data_source = new_ds
        else:
            self._iterator_fallback_skip_batches = self.iterator_start_batch
            self._iterator_fallback_max_batches = iterator_max_batches

    @property
    def iterator_logical_batch_index(self) -> Optional[int]:
        """Batch index including ``iterator_start_batch``; ``None`` before any yield."""
        if self._iterator_batch_in_pass < 0:
            return None
        return self.iterator_start_batch + self._iterator_batch_in_pass

    def __iter__(self):
        if getattr(self, "_iterator_empty", False):
            return _EmptyLoaderIter()
        # Match torch.utils.data.DataLoader.__iter__ persistence semantics.
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = _RawSamplerIterator(self._raw_dataloader_iterator(), self)
            else:
                self._iterator._reset(self)
            return self._iterator
        return _RawSamplerIterator(self._raw_dataloader_iterator(), self)

    def __len__(self) -> int:
        if getattr(self, "_iterator_empty", False):
            return 0
        if self._iterator_fallback_skip_batches or self._iterator_fallback_max_batches is not None:
            full = super().__len__()
            remain = max(0, full - int(self._iterator_fallback_skip_batches))
            lim = self._iterator_fallback_max_batches
            if lim is not None:
                remain = min(remain, int(lim))
            return remain
        return super().__len__()

    @property
    def iterator_batch_in_pass(self) -> int:
        """Batches yielded so far in the current ``for`` loop (``-1`` until the first batch)."""
        return self._iterator_batch_in_pass

    def _filter_homogeneous_without_x(self, out: SamplerOutput) -> Data:
        data = filter_data_without_x(
            self.data,
            out.node,
            out.row,
            out.col,
            out.edge,
            self.node_sampler.edge_permutation,
        )
        if "n_id" not in data:
            data.n_id = out.node
        if out.edge is not None and "e_id" not in data:
            edge = out.edge.to(torch.long)
            perm = self.node_sampler.edge_permutation
            data.e_id = perm[edge] if perm is not None else edge
        data.batch = out.batch
        data.num_sampled_nodes = out.num_sampled_nodes
        data.num_sampled_edges = out.num_sampled_edges
        if out.orig_row is not None and out.orig_col is not None:
            data._orig_edge_index = torch.stack(
                [out.orig_row, out.orig_col], dim=0
            )
        data.input_id = out.metadata[0]
        data.seed_time = out.metadata[1]
        data.batch_size = out.metadata[0].size(0)
        return data

    def _fetch_gids_into_data(self, data: Data) -> None:
        dev = self._gids_device
        if dev is None or self._gids is None or self._gids_feat_dim is None:
            raise RuntimeError("GIDS fetch requires gids, gids_device, and gids_feat_dim")
        idx = data.n_id.to(dev, dtype=torch.long)
        t0 = time.perf_counter()
        data.x = self._gids.fetch_feature(idx, self._gids_feat_dim)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if self._gids_timing_stats is not None:
            self._gids_timing_stats["s"] = float(self._gids_timing_stats["s"]) + (
                time.perf_counter() - t0
            )
            self._gids_timing_stats["n"] = int(self._gids_timing_stats["n"]) + int(
                idx.numel()
            )

    def filter_fn(self, out: Union[SamplerOutput, HeteroSamplerOutput]) -> Any:
        if self._gids is not None:
            if not isinstance(out, SamplerOutput):
                raise TypeError(
                    "gids=... expects homogeneous SamplerOutput from the sampler "
                    f"(got {type(out).__name__}). Hetero graphs are unsupported; "
                    "if this is homogeneous Data, filter_fn ran twice — use "
                    "filter_per_worker=False (set automatically when gids is set)."
                )
            if not isinstance(self.data, Data):
                raise TypeError(
                    "gids=... only supports homogeneous torch_geometric.data.Data."
                )
            if self.transform_sampler_output is not None:
                out = self.transform_sampler_output(out)
            batch = self._filter_homogeneous_without_x(out)
            self._fetch_gids_into_data(batch)
            if self.transform is not None:
                return self.transform(batch)
            return batch

        return super().filter_fn(out)
