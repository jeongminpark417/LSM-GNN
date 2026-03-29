#!/usr/bin/env python3
"""
PVP layout check with **64** PVP heads: one victim per buffer, meta idx == buffer id.

Phases:

0. **Fill** — ``fetch_feature`` rows **0..63** (64 cache lines, ``feat_dim == page_size/4``).
1. **Hints** — ``update_prefetch_timestamp`` on logical page **p** with **idx = p** (0..63) and
   **prefetch_ts** such that **reuse_val % nbuf == p** (``reuse_val`` is the high bits of
   ``next_reuse``, i.e. ``prefetch_ts``).  Device code uses **``cur_head = reuse_val % nbuf``**.
   For **p < 16** use **``prefetch_ts = p + nbuf``** so **reuse_val ≥ 16** and the
   **``reuse_val < 16``** early-deferral path in ``wb_find_slot_pvp`` is skipped; for **p ≥ 16**,
   **``prefetch_ts = p``** is enough.  You can drive **reuse_val** over time from the host with
   **``eviction_time_step``** when you extend the packing policy.
2. **Evict** — ``fetch_feature`` rows **64..127**, displacing the first 64 lines.
3. **Dump + verify** — Expect **64** occupied slots, **one per buffer**, ``rank_in_meta == buffer_idx``,
   and pinned embedding matches ``igb[buffer_idx]``. On success prints **Test Passed**.

**IGB:** ``--igb-npy`` is required; need at least **128** rows and ``emb_dim >= page_size//4``.

Example::

    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python example_pvp_buffer_walk.py --lsm-build .../lsm_module/build \\
        --igb-npy /path/to/igb.npy
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LSM_NVME_PYG = os.path.normpath(os.path.join(_HERE, ".."))
if _LSM_NVME_PYG not in sys.path:
    sys.path.insert(0, _LSM_NVME_PYG)

import numpy as np
import torch

from lsm_nvme_client import LSM_NVMeFeatureClient, unpack_prefetch_idx

N_FILL = 64
N_PVP_DEFAULT = 64


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PVP test: 64 buffers, prefetch_ts spreads heads, idx=0..63, verify vs IGB"
    )
    parser.add_argument("--lsm-build", type=str, default=None)
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1,
        help="Page-cache capacity in GiB (default 1; need enough lines for fill+evict)",
    )
    parser.add_argument(
        "--num-pvp-buffers",
        type=int,
        default=N_PVP_DEFAULT,
        help="PVP heads; default 64 (required for strict Test Passed)",
    )
    parser.add_argument(
        "--pvp-queue-depth",
        type=int,
        default=N_FILL,
        help="Slots per head; default 64",
    )
    parser.add_argument(
        "--igb-npy",
        type=str,
        required=True,
        help="Path to igb.npy [num_nodes, emb_dim], emb_dim >= page_size/4",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="np.allclose rtol for embedding check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="np.allclose atol for embedding check",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    page_size = int(args.page_size)
    elems_per_page = page_size // 4
    feat_dim = elems_per_page

    path = os.path.expanduser(args.igb_npy)
    if not os.path.isfile(path):
        print(f"error: --igb-npy not a file: {path}", file=sys.stderr)
        sys.exit(1)
    igb_arr = np.load(path, mmap_mode="r")
    if igb_arr.ndim != 2:
        print("error: igb.npy must be 2D [num_nodes, emb_dim]", file=sys.stderr)
        sys.exit(1)
    n_igb, d_igb = int(igb_arr.shape[0]), int(igb_arr.shape[1])
    if d_igb < feat_dim:
        print(
            f"error: igb emb_dim {d_igb} < floats/page {feat_dim} "
            f"(need full {feat_dim}-wide rows for one node per 4 KiB page)",
            file=sys.stderr,
        )
        sys.exit(1)
    if n_igb < 128:
        print("error: need at least 128 nodes (rows 0..127)", file=sys.stderr)
        sys.exit(1)

    num_ele = n_igb * feat_dim
    num_pvp_buffers = max(1, int(args.num_pvp_buffers))
    pvp_queue_depth = max(1, int(args.pvp_queue_depth))

    client = LSM_NVMeFeatureClient(
        page_size=page_size,
        off=0,
        cache_dim=feat_dim,
        num_ele=num_ele,
        num_ssd=1,
        cache_size=int(args.cache_size),
        ctrl_idx=args.device,
        lsm_build=args.lsm_build,
        repo_root=args.repo_root,
        is_pvp=True,
        num_pvp_buffers=num_pvp_buffers,
        pvp_queue_depth=pvp_queue_depth,
    )
    store = client._store

    rows_per_backing_page = elems_per_page // feat_dim
    if rows_per_backing_page != 1:
        print("error: internal layout expects one row per page", file=sys.stderr)
        sys.exit(1)

    # --- Phase 0: fill cache with rows 0..63 ---
    print("=== phase 0: fetch rows 0..63 (fill 64 lines) ===", flush=True)
    rows_a = torch.arange(N_FILL, device=device, dtype=torch.int64)
    _ = client.fetch_feature(rows_a, feat_dim)
    torch.cuda.synchronize()

    # --- Phase 1: reuse_val % nbuf == p => cur_head == p (wb_find_slot_pvp); reuse_val >= 16 for p<16 ---
    nbuf = num_pvp_buffers
    logical_pages = torch.arange(N_FILL, device=device, dtype=torch.int64)
    pages_i = torch.arange(N_FILL, device=device, dtype=torch.int32)
    prefetch_ts = torch.where(
        pages_i < 16,
        pages_i + int(nbuf),
        pages_i,
    )
    print(
        "=== phase 1: update_prefetch_timestamp prefetch_ts=p (or p+nbuf if p<16), idx=p ===",
        flush=True,
    )
    node_idx = torch.arange(N_FILL, device=device, dtype=torch.int64)
    client.update_prefetch_timestamp(logical_pages, prefetch_ts, node_idx)
    torch.cuda.synchronize()

    store.eviction_time_step = 0
    store.eviction_head_ptr = 0

    # --- Phase 2: evict with rows 64..127 ---
    print("=== phase 2: fetch rows 64..127 (evict 0..63) ===", flush=True)
    rows_b = torch.arange(N_FILL, 2 * N_FILL, device=device, dtype=torch.int64)
    _ = client.fetch_feature(rows_b, feat_dim)
    torch.cuda.synchronize()

    # --- Phase 3: dump + verify ---
    print("=== phase 3: PVP dump (meta idx should equal evicted node 0..63) ===", flush=True)
    print(
        "Legend: rank_in_meta = idx 0..63. PVP head = reuse_val % nbuf (reuse_val = prefetch_ts). "
        "Expect buffer_idx == rank_in_meta; emb matches igb[rank_in_meta].",
        flush=True,
    )
    print(flush=True)
    counts = np.asarray(client.pvp_copy_device_queue_counts(), dtype=np.uint32)
    meta = np.asarray(client.pvp_copy_host_meta_ids(), dtype=np.uint64)
    emb = np.asarray(client.pvp_copy_host_embeddings(), dtype=np.float32)
    if counts.size == 0 or meta.size == 0 or emb.size == 0:
        print("error: empty PVP snapshot", file=sys.stderr)
        sys.exit(1)

    def print_slot(rank: int, h: int, s: int, vec: np.ndarray) -> None:
        d8 = min(8, feat_dim)
        pin8 = np.asarray(vec[:d8], dtype=np.float32)
        n_in_buf = int(counts[h])
        line = (
            f"  buffer_idx={h} elems_in_buffer={n_in_buf} "
            f"rank_in_meta={rank} depth_slot={s} pinned_emb_first{d8}={pin8}"
        )
        nid = int(rank)
        if 0 <= nid < n_igb:
            ref8 = np.asarray(igb_arr[nid, :d8], dtype=np.float32)
            line += f" igb_row={nid} igb_npy_emb_first{d8}={ref8}"
        else:
            line += f" igb_row={nid} (oob)"
        print(line, flush=True)
        print(flush=True)

    records: list[tuple[int, int, int, np.ndarray]] = []
    total_slots = 0
    for h in range(int(counts.shape[0])):
        c = int(counts[h])
        total_slots += c
        for s in range(c):
            raw = int(meta[h, s]) % (1 << 64)
            rank = int(unpack_prefetch_idx(raw))
            vec = emb[h, s, :feat_dim]
            records.append((rank, h, s, vec))

    print(
        f"occupied_slots={total_slots} heads={num_pvp_buffers} depth={pvp_queue_depth} "
        f"feat_dim={feat_dim} floats_per_page={elems_per_page}",
        flush=True,
    )
    print(flush=True)
    print("--- A) Grouped by buffer_idx ---", flush=True)
    print(flush=True)
    for h in range(int(counts.shape[0])):
        c = int(counts[h])
        if c == 0:
            continue
        print(f"  --- buffer_idx={h}  elems_in_buffer={c} ---", flush=True)
        for s in range(c):
            raw = int(meta[h, s]) % (1 << 64)
            rank = int(unpack_prefetch_idx(raw))
            vec = emb[h, s, :feat_dim]
            print_slot(rank, h, s, vec)

    print("--- B) Sorted by rank_in_meta ---", flush=True)
    print(flush=True)
    records.sort(key=lambda t: (t[0], t[1], t[2]))
    for rank, h, s, vec in records:
        print_slot(rank, h, s, vec)

    # --- Strict check: 64 buffers, 1 slot each, idx == buffer id, emb == igb[idx] ---
    ok = True
    fail_msgs: list[str] = []
    if num_pvp_buffers != N_FILL:
        ok = False
        fail_msgs.append(
            f"num_pvp_buffers={num_pvp_buffers} (expected {N_FILL} for strict layout test)"
        )
    if total_slots != N_FILL:
        ok = False
        fail_msgs.append(f"total occupied slots={total_slots} (expected {N_FILL})")

    n_heads = int(counts.shape[0])
    for h in range(num_pvp_buffers):
        if h >= n_heads:
            ok = False
            fail_msgs.append(f"buffer_idx={h}: missing queue count (only {n_heads} heads)")
            continue
        c = int(counts[h])
        if c != 1:
            ok = False
            fail_msgs.append(f"buffer_idx={h}: elems_in_buffer={c} (expected 1)")
            continue
        raw = int(meta[h, 0]) % (1 << 64)
        rid = int(unpack_prefetch_idx(raw))
        if rid != h:
            ok = False
            fail_msgs.append(
                f"buffer_idx={h}: rank_in_meta={rid} (expected {h}, same as buffer id)"
            )
        pin = np.asarray(emb[h, 0, :feat_dim], dtype=np.float32)
        ref = np.asarray(igb_arr[h, :feat_dim], dtype=np.float32)
        if not np.allclose(pin, ref, rtol=args.rtol, atol=args.atol):
            ok = False
            fail_msgs.append(f"buffer_idx={h}: embedding mismatch vs igb row {h}")

    if ok and num_pvp_buffers == N_FILL:
        print("Test Passed", flush=True)
    else:
        print("Test Failed", flush=True)
        for m in fail_msgs:
            print(f"  - {m}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
