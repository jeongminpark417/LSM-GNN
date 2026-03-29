#!/usr/bin/env python3
"""
PyG training on **IGB / OGB** homogeneous graphs using **LSM_NVMe** (no GIDS).

Same layout as ``../train_bam_pyg_igb.py``, but features come from
:class:`lsm_nvme_client.LSM_NVMeFeatureClient` and neighbor sampling uses
``lsm_nvme_pyg.neighbor_loader_lsm_nvme.LSM_GNN_Neighbor_Loader`` with
``lsm_nvme=`` kwargs.

Example::

    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python train_bam_pyg_igb_lsm_nvme.py --bam 1 --data IGB --dataset_size small ...
    python train_bam_pyg_igb_lsm_nvme.py --bam 1 --pvp ...   # PVP path + batch prefetch (see --pvp-num-buffers)
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
_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
# ../pyg has dataloader_pyg, train_bam_pyg, and a *different* lsm_gnn_neighbor_loader (gids=).
# This directory must be ahead of _PARENT so `lsm_gnn_neighbor_loader` is the NVMe copy.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
else:
    try:
        sys.path.remove(_HERE)
    except ValueError:
        pass
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from dataloader_pyg import build_homogeneous_pyg_data
from pyg_loader_skip_features import data_without_dense_node_features
from train_bam_pyg import (
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

from neighbor_loader_lsm_nvme import LSM_GNN_Neighbor_Loader
from lsm_nvme_client import LSM_NVMeFeatureClient


def make_igb_neighbor_loaders(
    data,
    num_neighbors: list[int],
    batch_size: int,
    bam_loader_kw: dict,
):
    """Train/val/test :class:`LSM_GNN_Neighbor_Loader` instances (LSM_NVMe aliases)."""
    common = {
        "num_neighbors": num_neighbors,
        "batch_size": batch_size,
        "num_workers": 0,
        **bam_loader_kw,
    }
    train_loader = LSM_GNN_Neighbor_Loader(
        data,
        input_nodes=data.train_mask,
        shuffle=True,
        **common,
    )
    val_loader = LSM_GNN_Neighbor_Loader(
        data,
        input_nodes=data.val_mask,
        shuffle=False,
        **common,
    )
    test_loader = LSM_GNN_Neighbor_Loader(
        data,
        input_nodes=data.test_mask,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="PyG GraphSAGE on IGB/OGB + optional LSM_NVMe (no GIDS)"
    )
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
        help="lsm_module CMake build dir (must contain LSM_NVMe/)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap train/val/test to N mini-batches per epoch (smoke/debug). "
            "Omit for a full pass each epoch; use 0 for full pass if passed explicitly."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-neighbors", type=str, default="10,15")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bam", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--pvp",
        action="store_true",
        help="Enable LSM_NVMe PVP (is_pvp) and neighbor-loader batch prefetch queue (requires --bam 1).",
    )
    parser.add_argument(
        "--pvp-num-buffers",
        type=int,
        default=64,
        help="PVP device num_pvp_buffers / queue depth and loader lookahead depth (default 64).",
    )
    parser.add_argument("--num-ssd", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=10)
    parser.add_argument(
        "--num-ele",
        type=int,
        default=None,
        help="BAM backing-store element count (default: IGB/OGB presets)",
    )
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=None,
        help="Override input dim for model (default: data.x.size(1))",
    )
    parser.add_argument(
        "--full-batch",
        action="store_true",
        help="Full-graph training (no NeighborLoader / pyg-lib)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU if CUDA is unavailable (default: exit with error).",
    )
    args = parser.parse_args()

    if args.pvp and not args.bam:
        print("error: --pvp requires --bam 1 (LSM_NVMe PVP path).", file=sys.stderr)
        sys.exit(1)
    if args.pvp_num_buffers <= 0:
        print("error: --pvp-num-buffers must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.max_batches is None or int(args.max_batches) <= 0:
        max_batches = None
    else:
        max_batches = int(args.max_batches)

    if args.num_ele is None:
        if args.data == "IGB":
            args.num_ele = 300 * 1000 * 1000 * 1024
        else:
            args.num_ele = 111059956 * 128 * 2

    torch.manual_seed(args.seed)
    cuda_ok = torch.cuda.is_available()
    if args.bam and not cuda_ok:
        print("error: --bam 1 requires CUDA.", file=sys.stderr)
        sys.exit(1)
    if not cuda_ok and not args.allow_cpu:
        print(
            "error: CUDA is not available; refusing to run on CPU.\n"
            "  Pass --allow-cpu to override.",
            file=sys.stderr,
        )
        sys.exit(1)
    device = torch.device(f"cuda:{args.device}" if cuda_ok else "cpu")
    print(
        f"torch.cuda.is_available()={cuda_ok}  ->  using device: {device}",
        flush=True,
    )

    print("Loading graph (this can take a while)...", flush=True)
    t_load = time.time()
    data = build_homogeneous_pyg_data(args)
    print(
        f"Loaded in {time.time() - t_load:.1f}s | nodes={data.num_nodes} "
        f"| edges={data.edge_index.size(1)} | x={tuple(data.x.shape)}",
        flush=True,
    )

    ei_max = int(data.edge_index.max().item())
    if ei_max >= data.num_nodes:
        print(
            f"error: edge_index references node id {ei_max} but num_nodes={data.num_nodes}",
            file=sys.stderr,
        )
        sys.exit(1)
    num_classes = int(args.num_classes)
    y_min = int(data.y.min().item())
    y_max = int(data.y.max().item())
    if y_min < 0 or y_max >= num_classes:
        print(
            f"error: labels out of range for num_classes={num_classes} (min={y_min}, max={y_max})",
            file=sys.stderr,
        )
        sys.exit(1)

    feat_dim = args.feat_dim if args.feat_dim is not None else int(data.x.size(1))

    if args.bam:
        data = data_without_dense_node_features(data)
        print(
            f"Slim BAM Data: x shape {tuple(data.x.shape)} (features via LSM_NVMe only; "
            "no full-dim loader collation/H2D)",
            flush=True,
        )

    gids = None
    if args.bam:
        if args.data == "OGB" and args.page_size == 4096:
            args.page_size = 128 * 4
        lsm_build = args.lsm_build or os.path.join(args.repo_root, "lsm_module", "build")
        nvme_pkg = os.path.join(lsm_build, "LSM_NVMe")
        _prepend_path(nvme_pkg)
        nb = int(args.pvp_num_buffers)
        gids = LSM_NVMeFeatureClient(
            page_size=args.page_size,
            off=0,
            cache_dim=feat_dim,
            num_ele=args.num_ele,
            num_ssd=args.num_ssd,
            cache_size=args.cache_size,
            ctrl_idx=args.device,
            no_init=False,
            lsm_build=lsm_build,
            repo_root=args.repo_root,
            is_pvp=bool(args.pvp),
            num_pvp_buffers=nb if args.pvp else 0,
            pvp_queue_depth=nb if args.pvp else 0,
        )
        if args.pvp:
            print(
                f"LSM_NVMe PVP: is_pvp=True  num_pvp_buffers=pvp_queue_depth={nb}  "
                f"page cache --cache-size={args.cache_size} GiB",
                flush=True,
            )

    train_loader = val_loader = test_loader = None
    bam_feat_stats: dict[str, float | int] | None = None
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
        bam_loader_kw: dict = {}
        if args.bam:
            if gids is None:
                print("error: --bam 1 but LSM_NVMe client failed to initialize.", file=sys.stderr)
                sys.exit(1)
            bam_feat_stats = {"s": 0.0, "n": 0}
            bam_loader_kw = {
                "lsm_nvme": gids,
                "lsm_nvme_feat_dim": feat_dim,
                "lsm_nvme_device": device,
                "lsm_nvme_timing_stats": bam_feat_stats,
            }
            if args.pvp:
                bam_loader_kw["pvp_batch_prefetch"] = True
                bam_loader_kw["num_pvp_buffers"] = int(args.pvp_num_buffers)

        print(
            "Dataloader: LSM_GNN_Neighbor_Loader (lsm_nvme_pyg/neighbor_loader_lsm_nvme.py).",
            flush=True,
        )
        if args.bam and args.pvp:
            print(
                f"NeighborLoader: pvp_batch_prefetch=True  num_pvp_buffers={args.pvp_num_buffers}",
                flush=True,
            )
        if max_batches is not None:
            print(
                f"Limiting train/val/test to {max_batches} batch(es) per epoch "
                f"(omit --max-batches for a full epoch).",
                flush=True,
            )

        train_loader, val_loader, test_loader = make_igb_neighbor_loaders(
            data,
            num_neighbors,
            args.batch_size,
            bam_loader_kw,
        )

    from models_pyg import GraphSAGE

    model = GraphSAGE(
        feat_dim,
        args.hidden_channels,
        num_classes,
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
                data, model, optimizer, device, use_bam, gids, feat_dim
            )
            total_feat_fetch_s += fs
            total_n_feature_nodes += nf
            val_acc = eval_fullbatch(
                data, model, device, use_bam, gids, feat_dim, "val_mask"
            )
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}% "
                f"| train_batches=1 val_batches=1"
            )
        test_acc = eval_fullbatch(
            data, model, device, use_bam, gids, feat_dim, "test_mask"
        )
        n_te = 1
    else:
        for epoch in range(args.epochs):
            if bam_feat_stats is not None:
                bam_feat_stats["s"] = 0.0
                bam_feat_stats["n"] = 0
            loss, fs, nf, n_tr = train_epoch(
                train_loader,
                model,
                optimizer,
                device,
                use_bam,
                gids,
                feat_dim,
                bam_features_from_loader=use_bam,
                max_batches=max_batches,
            )
            if use_bam and bam_feat_stats is not None:
                total_feat_fetch_s += float(bam_feat_stats["s"])
                total_n_feature_nodes += int(bam_feat_stats["n"])
            else:
                total_feat_fetch_s += fs
                total_n_feature_nodes += nf
            val_acc, n_va = eval_epoch(
                val_loader,
                model,
                device,
                use_bam,
                gids,
                feat_dim,
                bam_features_from_loader=use_bam,
                max_batches=max_batches,
            )
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}% "
                f"| train_batches={n_tr} val_batches={n_va}"
            )
        test_acc, n_te = eval_epoch(
            test_loader,
            model,
            device,
            use_bam,
            gids,
            feat_dim,
            bam_features_from_loader=use_bam,
            max_batches=max_batches,
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
            feat_dim,
            wall_s=wall_s,
            ssd_read_ops=n_ssd,
            page_size=ps if n_ssd is not None else None,
        )


if __name__ == "__main__":
    main()
