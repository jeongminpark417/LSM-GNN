#!/usr/bin/env python3
"""Post-process NeighborLoader batches (hooks, prefetch, feature_fn).

The object yielded by ``for batch in loader`` is built **inside PyG** (sampler +
``filter_fn``). To customize **that** step, pass callbacks to
:class:`~torch_geometric.loader.NeighborLoader` itself:

* ``transform_sampler_output`` — mutates :class:`~torch_geometric.sampler.SamplerOutput`
  **before** subgraph features/edges are assembled into a :class:`~torch_geometric.data.Data`.
* ``transform`` — receives the final sampled :class:`~torch_geometric.data.Data` and
  must return the batch (runs **after** assembly, last step inside the loader).

This module does **not** replace those; it wraps the iterator to run optional hooks
and ``feature_fn`` **after** each batch leaves the inner ``NeighborLoader``.

For a **subclass** of ``NeighborLoader`` instead of a wrapper, use
``lsm_gnn_neighbor_loader.LSM_GNN_Neighbor_Loader``.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, Optional, TypeVar, Union

import torch

try:
    from torch_geometric.loader import NeighborLoader
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for pyg_neighbor_prefetch_loader"
    ) from e

T_batch = TypeVar("T_batch")
FeatureFn = Callable[[Any], Optional[torch.Tensor]]
BatchHook = Callable[[Any], None]


def iter_neighbor_batches_with_features(
    loader: NeighborLoader,
    feature_fn: Optional[FeatureFn] = None,
    *,
    after_sample_hook: Optional[BatchHook] = None,
    before_feature_hook: Optional[BatchHook] = None,
) -> Iterator[Any]:
    """
    Single-threaded iterator. Per batch:

    1. **Sampling** — ``next`` on ``loader`` (PyG neighbor sampling).
    2. ``after_sample_hook(batch)`` — optional (e.g. record time, enqueue work).
    3. ``before_feature_hook(batch)`` — optional (e.g. start custom prefetch).
    4. ``feature_fn(batch)`` — optional; if it returns a tensor, set ``batch.x``.

    Then ``yield batch``.
    """
    for batch in loader:
        if after_sample_hook is not None:
            after_sample_hook(batch)
        if before_feature_hook is not None:
            before_feature_hook(batch)
        if feature_fn is not None:
            x = feature_fn(batch)
            if x is not None:
                batch.x = x
        yield batch


class PrefetchNeighborLoader:
    """
    Wrap an existing ``NeighborLoader``.

    - **prefetch_depth == 0** — behaves like :func:`iter_neighbor_batches_with_features`.
    - **prefetch_depth >= 1** — a **daemon** thread pulls batches from the inner
      loader (sampling runs there ahead of the consumer). The consumer applies
      ``feature_fn`` then ``yield`` s.

    The inner ``NeighborLoader`` should typically use ``num_workers=0`` to avoid
    stacking two worker pools unless you know what you are doing.
    """

    def __init__(
        self,
        loader: NeighborLoader,
        *,
        prefetch_depth: int = 0,
        feature_fn: Optional[FeatureFn] = None,
        after_sample_hook: Optional[BatchHook] = None,
        before_feature_hook: Optional[BatchHook] = None,
    ) -> None:
        if prefetch_depth < 0:
            raise ValueError("prefetch_depth must be >= 0")
        self._loader = loader
        self.prefetch_depth = prefetch_depth
        self.feature_fn = feature_fn
        self.after_sample_hook = after_sample_hook
        self.before_feature_hook = before_feature_hook
        self._stop = threading.Event()
        self._queue: Optional[queue.Queue[Optional[Any]]] = None
        self._thread: Optional[threading.Thread] = None

    def __iter__(self) -> Iterator[Any]:
        self._stop.clear()
        if self.prefetch_depth == 0:
            yield from iter_neighbor_batches_with_features(
                self._loader,
                self.feature_fn,
                after_sample_hook=self.after_sample_hook,
                before_feature_hook=self.before_feature_hook,
            )
            return

        maxsize = max(1, self.prefetch_depth)
        self._queue = queue.Queue(maxsize=maxsize)

        def producer() -> None:
            try:
                for batch in self._loader:
                    if self._stop.is_set():
                        return
                    while not self._stop.is_set():
                        try:
                            self._queue.put(batch, timeout=0.05)
                            break
                        except queue.Full:
                            continue
            finally:
                try:
                    self._queue.put(None, timeout=1.0)
                except Exception:
                    pass

        self._thread = threading.Thread(target=producer, daemon=True, name="PrefetchNeighborLoader")
        self._thread.start()

        try:
            assert self._queue is not None
            while True:
                batch = self._queue.get()
                if batch is None:
                    break
                if self.after_sample_hook is not None:
                    self.after_sample_hook(batch)
                if self.before_feature_hook is not None:
                    self.before_feature_hook(batch)
                if self.feature_fn is not None:
                    x = self.feature_fn(batch)
                    if x is not None:
                        batch.x = x
                yield batch
        finally:
            self.close()

    def close(self) -> None:
        """Signal prefetch thread to stop and drain the queue (safe after early break)."""
        self._stop.set()
        q = self._queue
        if q is not None:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._thread = None
        self._queue = None

    def __len__(self) -> int:
        return len(self._loader)
