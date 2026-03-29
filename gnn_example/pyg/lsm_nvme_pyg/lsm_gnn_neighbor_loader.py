#!/usr/bin/env python3
"""PyG :class:`~torch_geometric.loader.NeighborLoader` with LSM NVMe features and optional PVP batch prefetch.

Self-contained module (does not import repo-root ``lsm_gnn_neighbor_loader``).

Pass ``lsm_nvme=`` (:class:`lsm_nvme_client.LSM_NVMeFeatureClient`), ``lsm_nvme_feat_dim``, ``lsm_nvme_device``.

**PVP batch prefetch** (``pvp_batch_prefetch=True``): lookahead queue of raw sampler outputs.
On the first consumer step, the iterator pulls ``min(num_pvp_buffers, total_batches)`` samples
into a queue. Each step pops one batch, optionally pulls one more from the underlying DataLoader
and enqueues it, then runs the usual ``filter_fn`` (structure + ``fetch_feature``).  When
``cur_yield_index + num_pvp_buffers >= total_batches``, no further pull/enqueue is done; the
yielded :class:`~torch_geometric.data.Data` has ``tail_batch=True``.

When ``lsm_nvme`` has ``is_pvp=True``, a device staging tensor is still allocated for
:meth:`run_pvp_prefetch` / ``PVP_prefetch``.

``num_pvp_buffers`` here is the **batch lookahead depth**, not the NVMe client's PVP head count
(though you may set them to the same value).

Optional ``iterator_start_batch`` / ``iterator_max_batches`` limit seed batches.
"""

from __future__ import annotations

import copy
import time
from collections import deque
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
    out = copy.copy(data)
    if "x" in out:
        del out["x"]
    _filter_node_store_skip_x(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def _eff_num_batches(loader: "LSM_GNN_Neighbor_Loader", base_len: int) -> int:
    skip = int(loader._iterator_fallback_skip_batches)
    lim = loader._iterator_fallback_max_batches
    remain = max(0, base_len - skip)
    if lim is not None:
        remain = min(remain, int(lim))
    return remain


class _EmptyLoaderIter:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __len__(self) -> int:
        return 0

    def _reset(self, loader: Any, first_iter: bool = False) -> None:
        pass


class _RawSamplerIterator:
    __slots__ = ("_base", "_loader", "_pending_skip", "_yielded", "_limit")

    def __init__(self, base: Any, loader: "LSM_GNN_Neighbor_Loader") -> None:
        self._base = base
        self._loader = loader
        self._pending_skip = int(loader._iterator_fallback_skip_batches)
        self._limit = loader._iterator_fallback_max_batches
        self._yielded = 0
        loader._iterator_batch_in_pass = -1
        loader._pvp_reset_time_step_for_iterator()

    def __iter__(self):
        return self

    def _reset(self, loader: Any, first_iter: bool = False) -> None:
        self._pending_skip = int(self._loader._iterator_fallback_skip_batches)
        self._limit = self._loader._iterator_fallback_max_batches
        self._yielded = 0
        self._loader._iterator_batch_in_pass = -1
        self._loader._pvp_reset_time_step_for_iterator()
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
        return _eff_num_batches(self._loader, len(self._base))


class _PvpBatchPrefetchRawIterator:
    """Pop from lookahead queue; pull another sample only while ``pulls_done < eff_total``."""

    __slots__ = (
        "_base",
        "_loader",
        "_pending_skip",
        "_yielded",
        "_limit",
        "_queue",
        "_eff_total",
        "_num_buf",
        "_warmup_done",
        "_pulls_done",
    )

    def __init__(self, base: Any, loader: "LSM_GNN_Neighbor_Loader") -> None:
        self._base = base
        self._loader = loader
        self._pending_skip = int(loader._iterator_fallback_skip_batches)
        self._limit = loader._iterator_fallback_max_batches
        self._yielded = 0
        self._queue: deque[Any] = deque()
        self._warmup_done = False
        self._pulls_done = 0
        self._num_buf = int(loader._pvp_batch_queue_size)
        self._eff_total = _eff_num_batches(loader, len(base))
        loader._iterator_batch_in_pass = -1
        loader._pvp_reset_time_step_for_iterator()
        loader._last_tail_batch = False

    def __iter__(self):
        return self

    def _reset(self, loader: Any, first_iter: bool = False) -> None:
        self._pending_skip = int(self._loader._iterator_fallback_skip_batches)
        self._limit = self._loader._iterator_fallback_max_batches
        self._yielded = 0
        self._queue.clear()
        self._warmup_done = False
        self._pulls_done = 0
        self._num_buf = int(self._loader._pvp_batch_queue_size)
        self._eff_total = _eff_num_batches(self._loader, len(self._base))
        self._loader._iterator_batch_in_pass = -1
        self._loader._pvp_reset_time_step_for_iterator()
        self._loader._last_tail_batch = False
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

        if not self._warmup_done:
            n0 = min(self._num_buf, self._eff_total)
            for _ in range(n0):
                try:
                    self._queue.append(next(self._base))
                    self._pulls_done += 1
                except StopIteration:
                    break
            self._warmup_done = True

        if not self._queue:
            raise StopIteration

        # No further graph sample/enqueue once we've already issued eff_total pulls.
        tail = self._pulls_done >= self._eff_total
        raw = self._queue.popleft()

        if self._pulls_done < self._eff_total:
            try:
                self._queue.append(next(self._base))
                self._pulls_done += 1
            except StopIteration:
                pass

        self._yielded += 1
        self._loader._iterator_batch_in_pass += 1
        self._loader._last_tail_batch = bool(tail)
        batch = self._loader.filter_fn(raw)
        if isinstance(batch, Data):
            batch.tail_batch = bool(tail)
        return batch

    def __len__(self) -> int:
        return self._eff_total


class LSM_GNN_Neighbor_Loader(NeighborLoader):
    """Neighbor sampling with ``lsm_nvme`` and optional PVP batch-queue prefetch."""

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
        lsm_nvme: Any = None,
        lsm_nvme_feat_dim: Optional[int] = None,
        lsm_nvme_device: Optional[torch.device] = None,
        lsm_nvme_timing_stats: Optional[dict[str, Union[float, int]]] = None,
        pvp_batch_prefetch: bool = False,
        num_pvp_buffers: int = 0,
        iterator_start_batch: int = 0,
        iterator_max_batches: Optional[int] = None,
        **kwargs,
    ) -> None:
        if pvp_batch_prefetch:
            if int(num_pvp_buffers) <= 0:
                raise ValueError("num_pvp_buffers must be > 0 when pvp_batch_prefetch=True")
            if lsm_nvme is None:
                raise ValueError("lsm_nvme is required when pvp_batch_prefetch=True")

        if lsm_nvme is not None:
            if lsm_nvme_feat_dim is None:
                raise ValueError("lsm_nvme_feat_dim is required when lsm_nvme is set")
            if lsm_nvme_device is None:
                raise ValueError("lsm_nvme_device is required when lsm_nvme is set")
            kwargs["filter_per_worker"] = False

        self._lsm_nvme = lsm_nvme
        self._lsm_nvme_feat_dim = int(lsm_nvme_feat_dim) if lsm_nvme_feat_dim is not None else None
        self._lsm_nvme_device = lsm_nvme_device
        self._lsm_nvme_timing_stats = lsm_nvme_timing_stats

        self._pvp_batch_prefetch = bool(pvp_batch_prefetch)
        self._pvp_batch_queue_size = int(num_pvp_buffers)
        self._last_tail_batch = False

        self._pvp_staging_buffer: Optional[torch.Tensor] = None
        self._pvp_time_step: int = 0
        self._pvp_prefetch_call_count: int = 0
        self._pvp_prefetch_embedding_count: int = 0
        self._pvp_embeddings_per_prefetch: int = 0

        if lsm_nvme is not None and bool(getattr(lsm_nvme, "is_pvp", False)):
            depth = int(getattr(lsm_nvme, "pvp_queue_depth", 0) or 0)
            ps = int(getattr(lsm_nvme, "page_size", 0) or 0)
            if depth > 0 and ps > 0:
                fpe = ps // 4
                dev = torch.device(str(lsm_nvme_device))
                self._pvp_staging_buffer = torch.empty(
                    (depth, fpe), dtype=torch.float32, device=dev
                )
                self._pvp_embeddings_per_prefetch = depth

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

    def _pvp_reset_time_step_for_iterator(self) -> None:
        if self._pvp_staging_buffer is not None:
            self._pvp_time_step = 0

    def _raw_dataloader_iterator(self):
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
        if self._iterator_batch_in_pass < 0:
            return None
        return self.iterator_start_batch + self._iterator_batch_in_pass

    def __iter__(self):
        if getattr(self, "_iterator_empty", False):
            return _EmptyLoaderIter()
        if self._pvp_batch_prefetch:
            it_cls = _PvpBatchPrefetchRawIterator
        else:
            it_cls = _RawSamplerIterator
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = it_cls(self._raw_dataloader_iterator(), self)
            else:
                self._iterator._reset(self)
            return self._iterator
        return it_cls(self._raw_dataloader_iterator(), self)

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
        return self._iterator_batch_in_pass

    @property
    def last_tail_batch(self) -> bool:
        """True if the last yielded batch used the tail path (no further sampler pull)."""
        return bool(self._last_tail_batch)

    @property
    def pvp_time_step(self) -> int:
        return int(self._pvp_time_step)

    @property
    def pvp_prefetch_call_count(self) -> int:
        return int(self._pvp_prefetch_call_count)

    @property
    def pvp_prefetch_embedding_count(self) -> int:
        return int(self._pvp_prefetch_embedding_count)

    @property
    def pvp_staging_buffer(self) -> Optional[torch.Tensor]:
        return self._pvp_staging_buffer

    def run_pvp_prefetch(self) -> None:
        if self._pvp_staging_buffer is None or self._lsm_nvme is None:
            raise RuntimeError(
                "run_pvp_prefetch requires lsm_nvme with is_pvp=True and valid PVP geometry"
            )
        if not getattr(self._lsm_nvme, "is_pvp", False):
            raise RuntimeError("run_pvp_prefetch requires LSM_NVMeFeatureClient with is_pvp=True")
        self._lsm_nvme.pvp_prefetch(self._pvp_staging_buffer, self._pvp_time_step)
        self._pvp_time_step += 1
        self._pvp_prefetch_call_count += 1
        self._pvp_prefetch_embedding_count += self._pvp_embeddings_per_prefetch

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

    def _fetch_lsm_nvme_into_data(self, data: Data) -> None:
        dev = self._lsm_nvme_device
        if dev is None or self._lsm_nvme is None or self._lsm_nvme_feat_dim is None:
            raise RuntimeError(
                "lsm_nvme fetch requires lsm_nvme, lsm_nvme_device, and lsm_nvme_feat_dim"
            )
        idx = data.n_id.to(dev, dtype=torch.long)
        t0 = time.perf_counter()
        data.x = self._lsm_nvme.fetch_feature(idx, self._lsm_nvme_feat_dim)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if self._lsm_nvme_timing_stats is not None:
            self._lsm_nvme_timing_stats["s"] = float(self._lsm_nvme_timing_stats["s"]) + (
                time.perf_counter() - t0
            )
            self._lsm_nvme_timing_stats["n"] = int(self._lsm_nvme_timing_stats["n"]) + int(
                idx.numel()
            )

    def filter_fn(self, out: Union[SamplerOutput, HeteroSamplerOutput]) -> Any:
        if self._lsm_nvme is not None:
            if not isinstance(out, SamplerOutput):
                raise TypeError(
                    "lsm_nvme=... expects homogeneous SamplerOutput from the sampler "
                    f"(got {type(out).__name__}). Hetero graphs are unsupported; "
                    "if this is homogeneous Data, filter_fn ran twice — use "
                    "filter_per_worker=False (set automatically when lsm_nvme is set)."
                )
            if not isinstance(self.data, Data):
                raise TypeError(
                    "lsm_nvme=... only supports homogeneous torch_geometric.data.Data."
                )
            if self.transform_sampler_output is not None:
                out = self.transform_sampler_output(out)
            batch = self._filter_homogeneous_without_x(out)
            self._fetch_lsm_nvme_into_data(batch)
            if self.transform is not None:
                return self.transform(batch)
            return batch

        return super().filter_fn(out)
