#!/usr/bin/env python3
"""
PyTorch Geometric training example with optional BAM / GIDS feature loads.

Mirrors the idea in ``gnn_example/gids_training.py``: with ``--bam 1``, node
features for each sampled subgraph come from ``GIDS.GIDS.fetch_feature`` (NVMe)
instead of ``data.x``.

Install (this example needs **PyTorch** and **PyG**; conda ``base`` often has neither)::

    # CPU wheels (simplest smoke test):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch-geometric

    # Or with Conda (pick the channel that matches your CUDA):
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install torch-geometric

Also ensure ``BAM_Feature_Store`` (from ``lsm_module`` build) and ``GIDS`` (from
``LSM_GNN_Setup``) are importable; use ``--lsm-build`` / ``--repo-root`` below.

``NeighborLoader`` needs **pyg-lib** or **torch-sparse** (not installed by
``pip install torch-geometric`` alone). The script prints a matching
``pip install ... -f https://data.pyg.org/whl/torch-...`` line if they are
missing, or use::

    python train_bam_pyg.py --epochs 2 --num-nodes 5000 --bam 0 --full-batch

By default the script **exits with an error** if CUDA is unavailable. For a tiny
CPU-only smoke test, add ``--allow-cpu``.

Synthetic demo (no NVMe; needs GPU or add ``--allow-cpu``)::

    python train_bam_pyg.py --epochs 2 --num-nodes 5000 --bam 0

With BAM (needs GPU, libnvm, BAM devices, and a backing store consistent with
the graph node ids you sample; the default synthetic graph is only for checking
the Python wiring against a matching test dataset)::

    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python train_bam_pyg.py --bam 1 --lsm-build /path/to/lsm_module/build

Note: ``gids_training.py`` relies on a **custom DGL DataLoader** (``bam=…``,
``bam_init``) in your DGL build. This directory uses **plain PyG**
``NeighborLoader`` and calls **GIDS** directly for features instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print(
        "error: PyTorch is not installed for this Python.\n"
        f"  interpreter: {sys.executable}\n"
        "  fix (CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  fix (GPU): see https://pytorch.org/get-started/locally/\n"
        "  then:      pip install torch-geometric",
        file=sys.stderr,
    )
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def _prepend_path(p: str) -> None:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _neighbor_sampler_backend_ok() -> bool:
    for mod in ("pyg_lib", "torch_sparse"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def _pyg_wheel_index_url() -> str:
    """Wheel index for pyg_lib / torch_sparse matching this torch build."""
    tv = torch.__version__
    if "+" in tv:
        a, b = tv.split("+", 1)
        suffix = f"{a}+{b}"
    else:
        suffix = f"{tv}+cpu"
    return f"https://data.pyg.org/whl/torch-{suffix}.html"


def build_adam_optimizer(parameters, lr: float, weight_decay: float):
    """
    Prefer foreach=False: multi-tensor Adam can surface cudaErrorUnknown on some
    driver/GPU stacks while the real fault was earlier (async CUDA).
    """
    try:
        return torch.optim.Adam(
            parameters, lr=lr, weight_decay=weight_decay, foreach=False
        )
    except TypeError:
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def _make_synthetic_data(num_nodes: int, feat_dim: int, num_classes: int, seed: int):
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    row = torch.randint(0, num_nodes, (num_nodes * 8,), dtype=torch.long)
    col = torch.randint(0, num_nodes, (num_nodes * 8,), dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_index = torch.unique(edge_index, dim=1)

    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    n_train = int(num_nodes * 0.6)
    n_val = int(num_nodes * 0.2)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    x = torch.randn(num_nodes, feat_dim)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def train_epoch(loader, model, optimizer, device, use_bam: bool, gids, feat_dim: int):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_bam:
            if not hasattr(batch, "n_id") or batch.n_id is None:
                raise RuntimeError(
                    "Batch missing n_id; use a recent PyG NeighborLoader for BAM gathers."
                )
            idx = batch.n_id.to(device, dtype=torch.long)
            x = gids.fetch_feature(idx, feat_dim)
        else:
            x = batch.x
        logits = model(x, batch.edge_index)[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.batch_size
        n += batch.batch_size
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(loader, model, device, use_bam: bool, gids, feat_dim: int):
    model.eval()
    correct = 0
    tot = 0
    for batch in loader:
        batch = batch.to(device)
        if use_bam:
            idx = batch.n_id.to(device, dtype=torch.long)
            x = gids.fetch_feature(idx, feat_dim)
        else:
            x = batch.x
        logits = model(x, batch.edge_index)[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum())
        tot += int(y.numel())
    return 100.0 * correct / max(tot, 1)


def train_epoch_fullbatch(data, model, optimizer, device, use_bam: bool, gids, feat_dim: int):
    """Full-graph forward; no pyg-lib / torch-sparse required."""
    model.train()
    optimizer.zero_grad()
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    if use_bam:
        idx = torch.arange(data.num_nodes, device=device, dtype=torch.long)
        x = gids.fetch_feature(idx, feat_dim)
    else:
        x = data.x.to(device)
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def eval_fullbatch(data, model, device, use_bam: bool, gids, feat_dim: int, mask_attr: str):
    model.eval()
    edge_index = data.edge_index.to(device)
    mask = getattr(data, mask_attr).to(device)
    y = data.y.to(device)
    if use_bam:
        idx = torch.arange(data.num_nodes, device=device, dtype=torch.long)
        x = gids.fetch_feature(idx, feat_dim)
    else:
        x = data.x.to(device)
    logits = model(x, edge_index)
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean() * 100.0)


def main():
    parser = argparse.ArgumentParser(description="PyG NeighborLoader + optional GIDS/BAM")
    parser.add_argument("--repo-root", type=str, default=_REPO_ROOT)
    parser.add_argument(
        "--lsm-build",
        type=str,
        default=None,
        help="lsm_module CMake build dir (BAM_Feature_Store). Default: <repo>/lsm_module/build",
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
    parser.add_argument("--wb-size", type=int, default=8)
    parser.add_argument("--wb-queue-size", type=int, default=131072)
    parser.add_argument("--cpu-agg", action="store_true")
    parser.add_argument("--cpu-agg-q-depth", type=int, default=0)
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
        print("error: --bam 1 requires CUDA (GIDS uses GPU tensors).", file=sys.stderr)
        sys.exit(1)
    if not cuda_ok and not args.allow_cpu:
        print(
            "error: CUDA is not available; refusing to run on CPU.\n"
            "  Fix the GPU driver / busy GPU (avoid sudo; check nvidia-smi), or pass --allow-cpu "
            "for a small CPU-only smoke test.",
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

    train_loader = val_loader = test_loader = None
    if args.full_batch:
        pass
    elif not _neighbor_sampler_backend_ok():
        url = _pyg_wheel_index_url()
        print(
            "error: NeighborLoader needs pyg-lib or torch-sparse.\n"
            f"  wheel index for your PyTorch ({torch.__version__}):\n"
            f"    {url}\n"
            "  install example:\n"
            f"    pip install pyg_lib torch_scatter torch_sparse torch_cluster -f {url}\n"
            "  or run without neighbor sampling:\n"
            "    python train_bam_pyg.py ... --full-batch",
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
        gids_setup = os.path.join(args.repo_root, "LSM_GNN_Setup")
        _prepend_path(lsm_build)
        _prepend_path(gids_setup)
        import GIDS  # noqa: E402

        gids = GIDS.GIDS(
            page_size=args.page_size,
            off=0,
            cache_dim=args.feat_dim,
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
    if args.full_batch:
        for epoch in range(args.epochs):
            loss = train_epoch_fullbatch(
                data, model, optimizer, device, use_bam, gids, args.feat_dim
            )
            val_acc = eval_fullbatch(
                data, model, device, use_bam, gids, args.feat_dim, "val_mask"
            )
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}%")
        test_acc = eval_fullbatch(
            data, model, device, use_bam, gids, args.feat_dim, "test_mask"
        )
    else:
        for epoch in range(args.epochs):
            loss = train_epoch(
                train_loader, model, optimizer, device, use_bam, gids, args.feat_dim
            )
            val_acc = eval_epoch(
                val_loader, model, device, use_bam, gids, args.feat_dim
            )
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val acc {val_acc:.2f}%")
        test_acc = eval_epoch(
            test_loader, model, device, use_bam, gids, args.feat_dim
        )
    print(f"Test acc {test_acc:.2f}% | wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
