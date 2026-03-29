#!/usr/bin/env python3
"""
Verify that GIDS / BaM NVMe backing store matches IGB ``node_feat.npy`` layout.

Loads node features the same way as ``dataloader_pyg.IGB260M`` (no full graph /
edge build). Fetches the first four node ids and four random node ids via
``GIDS.fetch_feature`` and compares to rows read from ``paper_feat``.

Prerequisites (same as ``train_bam_pyg_igb.py`` with ``--bam 1``)::

    export LD_LIBRARY_PATH=/path/to/bam/build/lib:$LD_LIBRARY_PATH

Example::

    python verify_gids_igb_embeddings.py \\
        --path /mnt/nvme0/bpark/IGB --dataset_size small --data IGB \\
        --device 0 --cache-size 1

Use ``--read-off`` if features on NVMe start at a non-zero byte offset (must
match ``readwrite`` ``--loffset`` and GIDS stride / page_size layout).
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"error: need numpy and torch ({e})", file=sys.stderr)
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataloader_pyg import IGB260M  # noqa: E402
from train_bam_pyg import _prepend_path  # noqa: E402

_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _compare_block(
    name: str,
    ref: torch.Tensor,
    got: torch.Tensor,
    rtol: float,
    atol: float,
) -> bool:
    """ref, got: same shape, float32."""
    ref_d = ref.detach()
    got_d = got.detach()
    diff = (ref_d - got_d).abs()
    max_err = float(diff.max().item())
    ok = torch.allclose(ref_d, got_d, rtol=rtol, atol=atol)
    print(f"\n=== {name} ===")
    print(f"  shape: {tuple(ref_d.shape)}  max |diff|: {max_err:.6g}  allclose: {ok}")
    if not ok:
        bad = int((diff > (atol + rtol * ref_d.abs())).sum().item())
        print(f"  mismatched elements (rough count): {bad}")
    return bool(ok)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare IGB node_feat.npy rows vs GIDS.fetch_feature (NVMe)"
    )
    p.add_argument("--path", type=str, default="/mnt/nvme14/IGB260M")
    p.add_argument(
        "--dataset_size",
        type=str,
        default="small",
        choices=["experimental", "small", "medium", "large", "full"],
    )
    p.add_argument("--in_memory", type=int, default=0, choices=[0, 1])
    p.add_argument("--uva_graph", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--num_classes", type=int, default=19, choices=[19, 2983, 171, 172, 173]
    )
    p.add_argument("--synthetic", type=int, default=0, choices=[0, 1])
    p.add_argument("--emb_size", type=int, default=1024)
    p.add_argument("--data", type=str, default="IGB", choices=["IGB", "OGB"])
    p.add_argument("--repo-root", type=str, default=_REPO_ROOT)
    p.add_argument(
        "--lsm-build",
        type=str,
        default=None,
        help="lsm_module CMake build dir (BAM_Feature_Store)",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-ssd", type=int, default=1)
    p.add_argument("--cache-size", type=int, default=10)
    p.add_argument(
        "--num-ele",
        type=int,
        default=None,
        help="Backing-store element count (floats); default: IGB/OGB presets",
    )
    p.add_argument("--wb-size", type=int, default=8)
    p.add_argument("--wb-queue-size", type=int, default=131072)
    p.add_argument("--cpu-agg", action="store_true")
    p.add_argument("--cpu-agg-q-depth", type=int, default=0)
    p.add_argument("--page-size", type=int, default=4096)
    p.add_argument(
        "--read-off",
        type=int,
        default=0,
        help="Byte offset on NVMe (GIDS ``off``); must match readwrite --loffset",
    )
    p.add_argument(
        "--cache-dim",
        type=int,
        default=None,
        help="Stride in floats per node on NVMe (default: feature dim from file)",
    )
    p.add_argument(
        "--rtol", type=float, default=1e-4, help="torch.allclose rtol (float32 IO)"
    )
    p.add_argument(
        "--atol", type=float, default=1e-5, help="torch.allclose atol (float32 IO)"
    )
    args = p.parse_args()

    if args.num_ele is None:
        if args.data == "IGB":
            args.num_ele = 300 * 1000 * 1000 * 1024
        else:
            args.num_ele = 111059956 * 128 * 2

    if args.data == "OGB" and args.page_size == 4096:
        args.page_size = 128 * 4

    if not torch.cuda.is_available():
        print("error: CUDA required for GIDS.fetch_feature.", file=sys.stderr)
        sys.exit(1)

    dev = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)

    print("Loading IGB/OGB node features (IGB260M paths, no edge tensors)...", flush=True)
    dataset = IGB260M(
        root=args.path,
        size=args.dataset_size,
        in_memory=args.in_memory,
        uva_graph=args.uva_graph,
        classes=args.num_classes,
        synthetic=args.synthetic,
        emb_size=args.emb_size,
        data=args.data,
    )
    feat_np = dataset.paper_feat
    num_nodes = int(feat_np.shape[0])
    feat_dim = int(feat_np.shape[1])
    cache_dim = int(args.cache_dim) if args.cache_dim is not None else feat_dim

    print(
        f"  num_nodes={num_nodes}  feat_dim={feat_dim}  cache_dim={cache_dim}  "
        f"page_size={args.page_size}  read_off={args.read_off}",
        flush=True,
    )
    if cache_dim < feat_dim:
        print(
            "error: cache_dim must be >= feat_dim (NVMe row stride vs columns to compare).",
            file=sys.stderr,
        )
        sys.exit(1)

    first = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(args.seed)
    rnd = torch.randint(low=0, high=num_nodes, size=(4,), generator=g, dtype=torch.long)
    print(f"  first ids: {first.tolist()}")
    print(f"  random ids (seed={args.seed}): {rnd.tolist()}")

    ref_first = torch.from_numpy(np.asarray(feat_np[first.numpy()])).float().cpu()
    ref_rnd = torch.from_numpy(np.asarray(feat_np[rnd.numpy()])).float().cpu()

    lsm_build = args.lsm_build or os.path.join(args.repo_root, "lsm_module", "build")
    gids_setup = os.path.join(args.repo_root, "LSM_GNN_Setup")
    _prepend_path(lsm_build)
    _prepend_path(gids_setup)

    try:
        import GIDS  # noqa: E402
    except ImportError as e:
        print(
            "error: cannot import GIDS. Set --lsm-build and LSM_GNN_Setup on PYTHONPATH.\n"
            f"  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    gids = GIDS.GIDS(
        page_size=args.page_size,
        off=args.read_off,
        cache_dim=cache_dim,
        num_ele=args.num_ele,
        num_ssd=args.num_ssd,
        cache_size=args.cache_size,
        wb_size=args.wb_size,
        wb_queue_size=args.wb_queue_size,
        cpu_agg=args.cpu_agg,
        cpu_agg_queue_size=args.cpu_agg_q_depth,
        ctrl_idx=args.device,
        no_init=False,
        ddp=False,
    )

    idx_first = first.to(dev)
    idx_rnd = rnd.to(dev)

    got_first = gids.fetch_feature(idx_first, feat_dim).float().cpu()
    torch.cuda.synchronize()
    got_rnd = gids.fetch_feature(idx_rnd, feat_dim).float().cpu()
    torch.cuda.synchronize()

    ok1 = _compare_block("first 4 nodes (ids 0..3)", ref_first, got_first, args.rtol, args.atol)
    ok2 = _compare_block("4 random nodes", ref_rnd, got_rnd, args.rtol, args.atol)

    if ok1 and ok2:
        print("\nPASS: file embeddings match GIDS for all checked nodes.")
        sys.exit(0)

    n_show = min(8, feat_dim)
    print("\n--- debug: first nodes 0..3, first {} floats (file vs GIDS) ---".format(n_show), flush=True)
    for row, nid in enumerate((0, 1, 2, 3)):
        print(f"\n  node id {nid}:", flush=True)
        print(f"    file (node_feat.npy): {ref_first[row, :n_show].tolist()}", flush=True)
        print(f"    GIDS (fetch_feature): {got_first[row, :n_show].tolist()}", flush=True)

    print("\nFAIL: mismatch — check NVMe contents, read_off, page_size, cache_dim, num_ele.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
