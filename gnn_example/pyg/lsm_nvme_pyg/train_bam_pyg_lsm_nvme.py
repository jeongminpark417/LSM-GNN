#!/usr/bin/env python3
"""
PyTorch Geometric training example with **LSM_NVMe** feature loads (no GIDS).

Same flow as ``../train_bam_pyg.py``: with ``--bam 1``, node features come from
:class:`lsm_nvme_client.LSM_NVMeFeatureClient` (``LSM_NVMe.LSM_NVMeStore.read_feature``)
instead of ``data.x``.

**Setup**

- Build the ``LSM_NVMe`` CMake target; the extension lives under
  ``<lsm_module_build>/LSM_NVMe/``.
- Set ``LD_LIBRARY_PATH`` to BAM ``libnvm`` (same as GIDS training).
- Default ``--lsm-build`` is ``<repo-root>/lsm_module/build``.

Example::

    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python train_bam_pyg_lsm_nvme.py --bam 1 --lsm-build /path/to/lsm_module/build
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import torch
except ImportError:
    print(
        "error: PyTorch is not installed for this Python.\n"
        f"  interpreter: {sys.executable}\n"
        "  fix: https://pytorch.org/get-started/locally/\n"
        "  then: pip install torch-geometric",
        file=sys.stderr,
    )
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from pyg_loader_skip_features import data_without_dense_node_features
from train_bam_pyg import (
    _make_synthetic_data,
    _neighbor_sampler_backend_ok,
    _prepend_path,
    _pyg_wheel_index_url,
    build_adam_optimizer,
    eval_epoch,
    eval_fullbatch,
    print_bam_feature_fetch_summary,
    train_epoch,
    train_epoch_fullbatch,
)

from lsm_nvme_client import LSM_NVMeFeatureClient


def main():
    parser = argparse.ArgumentParser(
        description="PyG NeighborLoader + LSM_NVMe (no GIDS/BAM_Feature_Store Python path)"
    )
    parser.add_argument("--repo-root", type=str, default=_REPO_ROOT)
    parser.add_argument(
        "--lsm-build",
        type=str,
        default=None,
        help="lsm_module CMake build dir (must contain LSM_NVMe/). Default: <repo>/lsm_module/build",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", type=str, default="10,10")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-nodes", type=int, default=5000)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--feat-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bam", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-ssd", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=10)
    parser.add_argument(
        "--num-ele",
        type=int,
        default=300 * 1000 * 1000 * 1024,
        help="BAM backing-store element count (match your dataset)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="BAM page size (bytes). For float32, feat_dim = page_size / 4.",
    )
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="Train on the full graph (no NeighborLoader; no pyg-lib/torch-sparse).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU-only run if CUDA is unavailable (default: exit with error).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cuda_ok = torch.cuda.is_available()
    if args.bam and not cuda_ok:
        print("error: --bam 1 requires CUDA (LSM_NVMe uses GPU tensors).", file=sys.stderr)
        sys.exit(1)
    if not cuda_ok and not args.allow_cpu:
        print(
            "error: CUDA is not available; refusing to run on CPU.\n"
            "  Pass --allow-cpu for a small CPU-only smoke test.",
            file=sys.stderr,
        )
        sys.exit(1)
    device = torch.device(f"cuda:{args.device}" if cuda_ok else "cpu")
    print(
        f"torch.cuda.is_available()={cuda_ok}  ->  using device: {device}",
        flush=True,
    )

    try:
        from torch_geometric.loader import NeighborLoader
    except ImportError:
        print(
            "error: torch-geometric is not installed.\n"
            "  pip install torch-geometric",
            file=sys.stderr,
        )
        sys.exit(1)

    data = _make_synthetic_data(
        args.num_nodes, args.feat_dim, args.num_classes, seed=args.seed
    )
    if args.bam:
        data = data_without_dense_node_features(data)
        print(
            f"Slim BAM Data: x shape {tuple(data.x.shape)} (LSM_NVMe-only features; "
            "NeighborLoader skips full-dim x)",
            flush=True,
        )

    train_loader = val_loader = test_loader = None
    if args.full_batch:
        pass
    elif not _neighbor_sampler_backend_ok():
        url = _pyg_wheel_index_url()
        print(
            "error: NeighborLoader needs pyg-lib or torch-sparse.\n"
            f"  {url}\n"
            f"  pip install pyg_lib torch_scatter torch_sparse torch_cluster -f {url}\n"
            "  or:  --full-batch",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        num_neighbors = [int(x) for x in args.num_neighbors.split(",") if x.strip()]
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
            num_workers=0,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            input_nodes=data.val_mask,
            shuffle=False,
            num_workers=0,
        )
        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            input_nodes=data.test_mask,
            shuffle=False,
            num_workers=0,
        )

    gids = None
    if args.bam:
        lsm_build = args.lsm_build or os.path.join(args.repo_root, "lsm_module", "build")
        nvme_pkg = os.path.join(lsm_build, "LSM_NVMe")
        _prepend_path(nvme_pkg)
        gids = LSM_NVMeFeatureClient(
            page_size=args.page_size,
            off=0,
            cache_dim=args.feat_dim,
            num_ele=args.num_ele,
            num_ssd=args.num_ssd,
            cache_size=args.cache_size,
            ctrl_idx=args.device,
            no_init=False,
            lsm_build=lsm_build,
            repo_root=args.repo_root,
        )

    from models_pyg import GraphSAGE

    model = GraphSAGE(
        args.feat_dim,
        args.hidden_channels,
        args.num_classes,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = build_adam_optimizer(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_bam = bool(args.bam)
    t0 = time.time()
    total_feat_fetch_s = 0.0
    total_n_feature_nodes = 0
    if args.full_batch:
        for epoch in range(args.epochs):
            loss, fs, nf = train_epoch_fullbatch(
                data, model, optimizer, device, use_bam, gids, args.feat_dim
            )
            total_feat_fetch_s += fs
            total_n_feature_nodes += nf
            val_acc = eval_fullbatch(
                data, model, device, use_bam, gids, args.feat_dim, "val_mask"
            )
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}% "
                f"| train_batches=1 val_batches=1"
            )
        test_acc = eval_fullbatch(
            data, model, device, use_bam, gids, args.feat_dim, "test_mask"
        )
        n_te = 1
    else:
        for epoch in range(args.epochs):
            loss, fs, nf, n_tr = train_epoch(
                train_loader, model, optimizer, device, use_bam, gids, args.feat_dim
            )
            total_feat_fetch_s += fs
            total_n_feature_nodes += nf
            val_acc, n_va = eval_epoch(
                val_loader, model, device, use_bam, gids, args.feat_dim
            )
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}% "
                f"| train_batches={n_tr} val_batches={n_va}"
            )
        test_acc, n_te = eval_epoch(
            test_loader, model, device, use_bam, gids, args.feat_dim
        )
    wall_s = time.time() - t0
    print(f"Test acc {test_acc:.2f}% | test_batches={n_te} | wall {wall_s:.1f}s")
    if use_bam:
        n_ssd = None
        ps = int(args.page_size)
        if gids is not None and hasattr(gids, "ssd_read_ops_count"):
            n_ssd = int(gids.ssd_read_ops_count())
        print_bam_feature_fetch_summary(
            total_feat_fetch_s,
            total_n_feature_nodes,
            args.feat_dim,
            wall_s=wall_s,
            ssd_read_ops=n_ssd,
            page_size=ps if n_ssd is not None else None,
        )


if __name__ == "__main__":
    main()
