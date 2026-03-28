#!/usr/bin/env python3
"""
PyG training on **IGB / OGB** homogeneous graphs (``dataloader_pyg.build_homogeneous_pyg_data``).

Same training loop and optional **GIDS/BAM** path as ``train_bam_pyg.py``, but the
graph comes from the same file layout as ``gnn_example/dataloader.py`` /
``gids_training.py``.

Example (small IGB partition, neighbor sampling)::

    python train_bam_pyg_igb.py --path /data/IGB260M --dataset_size small \\
        --data IGB --num_classes 19 --epochs 2 --batch-size 1024

Full graph on one GPU (only for small graphs; needs pyg-lib unless --full-batch)::

    python train_bam_pyg_igb.py ... --full-batch

With BAM (match ``num_ele`` / cache to your NVMe layout; see ``gids_training.py``)::

    export LD_LIBRARY_PATH=/path/to/bam/build/lib:$LD_LIBRARY_PATH
    python train_bam_pyg_igb.py --bam 1 --data IGB --dataset_size full ...
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
        "error: PyTorch is not installed.\n"
        f"  interpreter: {sys.executable}\n"
        "  install: https://pytorch.org/get-started/locally/",
        file=sys.stderr,
    )
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataloader_pyg import build_homogeneous_pyg_data
from train_bam_pyg import (
    _neighbor_sampler_backend_ok,
    _prepend_path,
    _pyg_wheel_index_url,
    eval_epoch,
    eval_fullbatch,
    train_epoch,
    train_epoch_fullbatch,
)

_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def main():
    parser = argparse.ArgumentParser(
        description="PyG GraphSAGE on IGB/OGB (homogeneous) + optional GIDS/BAM"
    )
    # Dataset (aligned with gids_training.py)
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/nvme14/IGB260M",
        help="Dataset root (IGB tree or OGB raw dir)",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="experimental",
        choices=["experimental", "small", "medium", "large", "full"],
    )
    parser.add_argument("--in_memory", type=int, default=0, choices=[0, 1])
    parser.add_argument("--uva_graph", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--num_classes", type=int, default=19, choices=[19, 2983, 171, 172, 173]
    )
    parser.add_argument("--synthetic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--emb_size", type=int, default=1024)
    parser.add_argument("--data", type=str, default="IGB", choices=["IGB", "OGB"])

    parser.add_argument("--repo-root", type=str, default=_REPO_ROOT)
    parser.add_argument(
        "--lsm-build",
        type=str,
        default=None,
        help="lsm_module CMake build dir (BAM_Feature_Store)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-neighbors", type=str, default="10,15")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bam", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-ssd", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=10)
    parser.add_argument(
        "--num-ele",
        type=int,
        default=None,
        help="BAM backing-store element count (default: IGB/OGB presets like gids_training)",
    )
    parser.add_argument("--wb-size", type=int, default=8)
    parser.add_argument("--wb-queue-size", type=int, default=131072)
    parser.add_argument("--cpu-agg", action="store_true")
    parser.add_argument("--cpu-agg-q-depth", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=None,
        help="Override input dim for model/BAM (default: data.x.size(1))",
    )
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="Full-graph training (no NeighborLoader / pyg-lib)",
    )
    args = parser.parse_args()

    if args.num_ele is None:
        if args.data == "IGB":
            args.num_ele = 300 * 1000 * 1000 * 1024
        else:
            args.num_ele = 111059956 * 128 * 2

    torch.manual_seed(args.seed)
    if args.bam and not torch.cuda.is_available():
        print("error: --bam 1 requires CUDA.", file=sys.stderr)
        sys.exit(1)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    try:
        from torch_geometric.loader import NeighborLoader
    except ImportError:
        print("error: pip install torch-geometric", file=sys.stderr)
        sys.exit(1)

    print("Loading graph (this can take a while)...", flush=True)
    t_load = time.time()
    data = build_homogeneous_pyg_data(args)
    print(
        f"Loaded in {time.time() - t_load:.1f}s | nodes={data.num_nodes} "
        f"| edges={data.edge_index.size(1)} | x={tuple(data.x.shape)}",
        flush=True,
    )

    feat_dim = args.feat_dim if args.feat_dim is not None else int(data.x.size(1))
    num_classes = int(args.num_classes)

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
        # Match gids_training.py: OGB uses page_size 128*4 (512) for float32 dim-128 rows.
        if args.data == "OGB" and args.page_size == 4096:
            args.page_size = 128 * 4
        lsm_build = args.lsm_build or os.path.join(args.repo_root, "lsm_module", "build")
        gids_setup = os.path.join(args.repo_root, "LSM_GNN_Setup")
        _prepend_path(lsm_build)
        _prepend_path(gids_setup)
        import GIDS  # noqa: E402

        gids = GIDS.GIDS(
            page_size=args.page_size,
            off=0,
            cache_dim=feat_dim,
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

    from models_pyg import GraphSAGE

    model = GraphSAGE(
        feat_dim,
        args.hidden_channels,
        num_classes,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_bam = bool(args.bam)
    t0 = time.time()
    if args.full_batch:
        for epoch in range(args.epochs):
            loss = train_epoch_fullbatch(
                data, model, optimizer, device, use_bam, gids, feat_dim
            )
            val_acc = eval_fullbatch(
                data, model, device, use_bam, gids, feat_dim, "val_mask"
            )
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}%")
        test_acc = eval_fullbatch(
            data, model, device, use_bam, gids, feat_dim, "test_mask"
        )
    else:
        for epoch in range(args.epochs):
            loss = train_epoch(
                train_loader, model, optimizer, device, use_bam, gids, feat_dim
            )
            val_acc = eval_epoch(
                val_loader, model, device, use_bam, gids, feat_dim
            )
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}%")
        test_acc = eval_epoch(test_loader, model, device, use_bam, gids, feat_dim)
    print(f"Test acc {test_acc:.2f}% | wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
