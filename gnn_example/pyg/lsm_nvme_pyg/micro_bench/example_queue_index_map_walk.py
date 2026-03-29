#!/usr/bin/env python3
"""
Microbench for ``build_node_queue_index_map``, ``index_map_add``, and ``index_map_remove``.

The cuco map lives on the GPU and is not exposed to Python; this script issues the same
calls on the device and maintains a **host-side mirror** of the documented semantics so we
can print the logical map after each step.

**Batch index column:** ``build_node_queue_index_map`` takes ``d_batch_idx_ptr`` (int32 per row):
the **0-based batch index** (batch1 → 0, batch2 → 1, …), i.e. which PVP buffer / lookahead batch
the row belongs to.

This script uses **four batches** of four nodes; every row from batch ``b`` gets
``queue_idx == b`` (same as ``num_pvp_buffers == 4`` heads)::

    batch 0 (buffer 0): nodes (1,2,3,4)   + queue_idx all 0
    batch 1 (buffer 1): nodes (1,2,3,5)   + queue_idx all 1
    batch 2 (buffer 2): nodes (2,5,6,7)   + queue_idx all 2
    batch 3 (buffer 3): nodes (3,7,8,9)   + queue_idx all 3

Then: ``index_map_add`` nodes ``(5,6,7,10)`` at time step **4**; ``index_map_remove``
``(1,2,3,4)`` at time step **0**; then ``index_map_remove`` ``(1,2,3,5)`` at time step **0**.

Example::

    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python example_queue_index_map_walk.py --lsm-build .../lsm_module/build
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LSM_NVME_PYG = os.path.normpath(os.path.join(_HERE, ".."))
if _LSM_NVME_PYG not in sys.path:
    sys.path.insert(0, _LSM_NVME_PYG)

import torch

from lsm_nvme_client import LSM_NVMeFeatureClient

INT32_MAX = 2147483647

# Four batches of nodes; queue_idx for each row = PVP buffer / batch id (0..3).
NODE_BATCHES = (
    (1, 2, 3, 4),
    (1, 2, 3, 5),
    (2, 5, 6, 7),
    (3, 7, 8, 9),
)


def _flatten_batches_with_buffer_id(
    node_batches: tuple[tuple[int, ...], ...],
) -> tuple[list[int], list[int]]:
    """One queue_idx per row: buffer id == batch index (like ``cur_head`` / PVP head)."""
    nodes: list[int] = []
    queues: list[int] = []
    for batch_id, nb in enumerate(node_batches):
        q = int(batch_id)
        for nid in nb:
            nodes.append(int(nid))
            queues.append(q)
    return nodes, queues


def host_build_node_queue_index_map(
    node_ids: list[int], queue_idx: list[int]
) -> dict[int, int]:
    """
    Mirror ``lsm_nvme_queue_map_build_impl`` / ``fill_node_reuse_map_entries_kernel``:
    sort by (node, batch_idx); per node segment, INT32_MAX if one row or all same batch index or
    second distinct index absent; else second-smallest distinct batch index.
    """
    pairs = sorted(zip(node_ids, queue_idx), key=lambda t: (t[0], t[1]))
    out: dict[int, int] = {}
    i = 0
    n = len(pairs)
    while i < n:
        nid = pairs[i][0]
        j = i + 1
        while j < n and pairs[j][0] == nid:
            j += 1
        seg = pairs[i:j]
        if len(seg) == 1:
            out[nid] = INT32_MAX
        else:
            anchor = seg[0][1]
            k = 1
            while k < len(seg) and seg[k][1] == anchor:
                k += 1
            out[nid] = INT32_MAX if k >= len(seg) else seg[k][1]
        i = j
    return out


def host_index_map_add(m: dict[int, int], batch: list[int], time_step: int) -> None:
    """Mirror device ``index_map_add``."""
    ts = int(time_step) & 0xFFFFFFFF
    if ts > 0x7FFFFFFF:
        ts = ts - 0x1_0000_0000
    for k in batch:
        if k < 0:
            continue
        if k not in m:
            m[k] = INT32_MAX
        elif m[k] == INT32_MAX:
            m[k] = ts


def host_index_map_remove(m: dict[int, int], batch: list[int], time_step: int) -> None:
    """Mirror device ``index_map_remove``."""
    ts = int(time_step) & 0xFFFFFFFF
    if ts > 0x7FFFFFFF:
        ts = ts - 0x1_0000_0000
    for k in batch:
        if k < 0:
            continue
        if k not in m:
            continue
        v = m[k]
        if v == INT32_MAX or v == ts:
            del m[k]


def format_map(m: dict[int, int]) -> str:
    parts = []
    for k in sorted(m.keys()):
        v = m[k]
        if v == INT32_MAX:
            parts.append(f"  node {k} -> INT32_MAX")
        else:
            parts.append(f"  node {k} -> {v}")
    return "\n".join(parts) if parts else "  (empty)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise queue index map build / add / remove (GPU + host mirror print)"
    )
    parser.add_argument("--lsm-build", type=str, default=None)
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--map-capacity",
        type=int,
        default=128,
        help="cuco static_map capacity lower bound (unique keys)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    flat_nodes, flat_queues = _flatten_batches_with_buffer_id(NODE_BATCHES)
    node_t = torch.tensor(flat_nodes, device=device, dtype=torch.int64)
    queue_t = torch.tensor(flat_queues, device=device, dtype=torch.int32)

    client = LSM_NVMeFeatureClient(
        page_size=4096,
        cache_dim=1024,
        num_ele=1 << 20,
        cache_size=1,
        ctrl_idx=args.device,
        lsm_build=args.lsm_build,
        repo_root=args.repo_root,
        no_init=True,
    )

    mirror: dict[int, int] = {}

    def print_step(title: str) -> None:
        print(flush=True)
        print(f"=== {title} ===", flush=True)
        print("(host mirror; same ops issued on GPU — map is not read back from device)", flush=True)
        print(format_map(mirror), flush=True)

    # --- Step 1: build from four batches ---
    print("Batches: node ids per batch; queue_idx = PVP buffer id (batch index):", flush=True)
    for bi, nb in enumerate(NODE_BATCHES):
        print(f"  batch / buffer {bi}: nodes={nb}  (queue_idx={bi} for each row)", flush=True)

    ts_build = 1
    mirror.clear()
    mirror.update(host_build_node_queue_index_map(list(flat_nodes), list(flat_queues)))
    client.build_node_queue_index_map(
        node_t, queue_t, int(args.map_capacity), int(ts_build) & 0xFFFFFFFF
    )
    torch.cuda.synchronize()
    print_step(f"After build_node_queue_index_map (time_step={ts_build})")

    # --- Step 2: index_map_add (5,6,7,10) @ 4 ---
    add_nodes = [5, 6, 7, 10]
    ts_add = 4
    host_index_map_add(mirror, add_nodes, ts_add)
    add_t = torch.tensor(add_nodes, device=device, dtype=torch.int64)
    client.index_map_add(add_t, int(ts_add) & 0xFFFFFFFF)
    torch.cuda.synchronize()
    print_step(f"After index_map_add {add_nodes} (time_step={ts_add})")

    # --- Step 3: index_map_remove (1,2,3,4) @ 0 ---
    rm_nodes = [1, 2, 3, 4]
    ts_rm = 0
    host_index_map_remove(mirror, rm_nodes, ts_rm)
    rm_t = torch.tensor(rm_nodes, device=device, dtype=torch.int64)
    client.index_map_remove(rm_t, int(ts_rm) & 0xFFFFFFFF)
    torch.cuda.synchronize()
    print_step(f"After index_map_remove {rm_nodes} (time_step={ts_rm})")

    # --- Step 4: index_map_remove (1,2,3,5) @ 0 ---
    rm_nodes_b = [1, 2, 3, 5]
    ts_rm = 1
    host_index_map_remove(mirror, rm_nodes_b, ts_rm)
    rm_b_t = torch.tensor(rm_nodes_b, device=device, dtype=torch.int64)
    client.index_map_remove(rm_b_t, int(ts_rm) & 0xFFFFFFFF)
    torch.cuda.synchronize()
    print_step(f"After index_map_remove {rm_nodes_b} (time_step={ts_rm})")

    print(flush=True)
    print("Microbench finished (GPU calls completed).", flush=True)


if __name__ == "__main__":
    main()
