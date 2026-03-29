#!/usr/bin/env python3
"""
Thin Python wrapper around ``LSM_NVMe.LSM_NVMeStore`` for PyG training.

Mirrors the subset of :class:`GIDS.GIDS` used by ``train_bam_pyg.py`` / neighbor
loaders: ``fetch_feature(index, dim)`` and ``gids_device`` / ``cache_dim`` for
compatibility with :class:`lsm_gnn_neighbor_loader.LSM_GNN_Neighbor_Loader``.

Also exposes :meth:`update_prefetch_timestamp` (device tensors) mapping to the
C++ extension.

**Import path:** add ``<lsm_module_build>/LSM_NVMe`` to ``sys.path`` (CMake places
the ``.so`` there). Set ``LD_LIBRARY_PATH`` to BAM ``libnvm`` like GIDS training.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# Matches ``lsm_gnn::next_reuse_not_resident_value()`` / readback when page not resident.
NEXT_REUSE_NOT_RESIDENT_U64: int = (1 << 64) - 1


def as_u64_from_signed_i64(x: int) -> int:
    """Interpret a PyTorch int64 element (possibly negative) as unsigned 64-bit."""
    return int(x) % (1 << 64)


def pack_prefetch_timestamp_idx(ts: int, idx: int) -> int:
    """
    Same packing as the CUDA ``update_prefetch_timestamp`` kernel: ``uint64_t``
    ``(ts << 48) | idx`` (idx lower 48 bits). The shift is truncated to 64 bits
    in C++; mask here so Python ``int`` matches device readback.
    """
    return (
        ((int(ts) & 0xFFFFFFFF) << 48) | (int(idx) & ((1 << 48) - 1))
    ) & ((1 << 64) - 1)


def unpack_prefetch_timestamp(packed_u64: int) -> int:
    return (int(packed_u64) % (1 << 64) >> 48) & 0xFFFFFFFF


def unpack_prefetch_idx(packed_u64: int) -> int:
    return int(packed_u64) % (1 << 64) & ((1 << 48) - 1)


def _prepend_path(p: str) -> None:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def import_lsm_nvme_module(*, lsm_build: Optional[str] = None, repo_root: Optional[str] = None):
    """Return the ``LSM_NVMe`` extension module (import after fixing ``sys.path``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = repo_root or os.path.normpath(os.path.join(here, "..", "..", ".."))
    build = lsm_build or os.path.join(root, "lsm_module", "build")
    pkg = os.path.join(build, "LSM_NVMe")
    _prepend_path(pkg)
    import LSM_NVMe  # noqa: E402

    return LSM_NVMe


class LSM_NVMeFeatureClient:
    """
    NVMe feature reads via ``lsm_nvme`` (``read_feature``), duck-compatible with
    ``GIDS.GIDS`` for ``fetch_feature`` / ``gids_device`` / ``cache_dim``.
    """

    def __init__(
        self,
        page_size: int = 4096,
        off: int = 0,
        cache_dim: int = 1024,
        num_ele: int = 300 * 1000 * 1000 * 1024,
        num_ssd: int = 1,
        cache_size: int = 10,
        ctrl_idx: int = 0,
        no_init: bool = False,
        lsm_build: Optional[str] = None,
        repo_root: Optional[str] = None,
        is_pvp: bool = False,
        num_pvp_buffers: int = 0,
        pvp_queue_depth: int = 0,
    ):
        self.cache_dim = int(cache_dim)
        self.gids_device = f"cuda:{int(ctrl_idx)}"
        self.is_pvp = bool(is_pvp)

        LSM_NVMe = import_lsm_nvme_module(lsm_build=lsm_build, repo_root=repo_root)
        self._store = LSM_NVMe.LSM_NVMeStore()
        self._store.cudaDevice = int(ctrl_idx)

        if not no_init:
            self._store.init_controllers(
                int(page_size),
                int(off),
                int(cache_size),
                int(num_ele),
                int(num_ssd),
                bool(is_pvp),
                int(num_pvp_buffers),
                int(pvp_queue_depth),
            )

    @property
    def num_pvp_buffers(self) -> int:
        return int(self._store.num_pvp_buffers)

    @property
    def pvp_queue_depth(self) -> int:
        return int(self._store.pvp_queue_depth)

    @property
    def page_size(self) -> int:
        return int(self._store.pageSize)

    def ssd_read_ops_count(self) -> int:
        """Cumulative NVMe read commands since init (see embedded page cache ``ssd_read_ops``)."""
        return int(self._store.ssd_read_ops_count())

    def fetch_feature(self, index: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Same contract as ``GIDS.GIDS.fetch_feature`` (CUDA ``index``, float32 out).
        If the client was constructed with ``is_pvp=True`` (PVP), the underlying
        ``read_feature`` uses the pinned eviction / prefetch path.
        """
        if index.device.type != "cuda":
            raise ValueError("LSM_NVMeFeatureClient.fetch_feature expects CUDA index tensor")
        index_ptr = index.data_ptr()
        index_size = int(index.numel())
        out = torch.zeros(
            (index_size, int(dim)),
            dtype=torch.float32,
            device=self.gids_device,
        )
        self._store.read_feature(
            out.data_ptr(), index_ptr, index_size, int(dim), self.cache_dim
        )
        return out

    def update_prefetch_timestamp(
        self,
        pages: torch.Tensor,
        timestamps: torch.Tensor,
        idxs: torch.Tensor,
    ) -> None:
        """
        Device tensors of equal length: logical page id (int64), timestamp (int32),
        idx payload (int64) for ``atomicMin`` on cache-line ``next_reuse``.
        """
        if pages.device.type != "cuda":
            raise ValueError("pages must be on CUDA")
        n = int(pages.numel())
        if int(timestamps.numel()) != n or int(idxs.numel()) != n:
            raise ValueError("pages, timestamps, idxs must have the same number of elements")
        pages = pages.reshape(-1).to(device=pages.device, dtype=torch.int64)
        timestamps = timestamps.reshape(-1).to(device=pages.device, dtype=torch.int32)
        idxs = idxs.reshape(-1).to(device=pages.device, dtype=torch.int64)
        self._store.update_prefetch_timestamp(
            pages.data_ptr(), timestamps.data_ptr(), idxs.data_ptr(), n
        )

    def read_next_reuse_for_pages(self, logical_pages: torch.Tensor) -> torch.Tensor:
        """
        Device int64 tensor of logical page indices. Returns int64 tensor of the
        same shape with raw ``next_reuse`` (unsigned 64-bit); use
        ``next_reuse_not_resident_u64()`` and :func:`unpack_prefetch_timestamp` /
        :func:`unpack_prefetch_idx` to interpret.
        """
        if logical_pages.device.type != "cuda":
            raise ValueError("logical_pages must be on CUDA")
        pages = logical_pages.reshape(-1).to(device=logical_pages.device, dtype=torch.int64)
        n = int(pages.numel())
        out = torch.empty(n, dtype=torch.int64, device=logical_pages.device)
        self._store.read_next_reuse_for_pages(pages.data_ptr(), out.data_ptr(), n)
        return out.reshape_as(logical_pages)

    def pvp_copy_device_queue_counts(self):
        """Device-side occupancy per PVP head (uint32 numpy array). Requires PVP init."""
        return self._store.pvp_copy_device_queue_counts()

    def pvp_copy_host_meta_ids(self):
        """Pinned host meta ring ``[num_buffers, depth]`` uint64 (copy)."""
        return self._store.pvp_copy_host_meta_ids()

    def pvp_copy_host_embeddings(self):
        """Pinned host victim feature bytes as ``[num_buffers, depth, page_size//4]`` float32 (copy)."""
        return self._store.pvp_copy_host_embeddings()

    def pvp_prefetch(self, dst: torch.Tensor, time_step: int) -> None:
        """
        ``PVP_prefetch``: copy pinned PVP ring for head ``time_step % num_pvp_buffers`` to
        ``dst`` (CUDA, contiguous).  ``dst`` must hold at least ``pvp_queue_depth * cache_dim``
        float32 elements (same layout as one row of ``pvp_copy_host_embeddings``).
        """
        if dst.device.type != "cuda":
            raise ValueError("pvp_prefetch expects a CUDA tensor")
        if not dst.is_contiguous():
            raise ValueError("pvp_prefetch expects a contiguous tensor")
        self._store.PVP_prefetch(dst.data_ptr(), int(time_step) & 0xFFFFFFFF)
