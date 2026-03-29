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

With ``--bam 1``, ``Data.x`` is replaced by a **slim** zero-column tensor so
``NeighborLoader`` does not copy full embeddings to each batch; only
``fetch_feature`` supplies features.

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

from pyg_loader_skip_features import data_without_dense_node_features


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


def print_bam_feature_fetch_summary(
    feat_agg_s: float,
    n_feat_nodes: int,
    feat_dim: int,
    prefix: str = "",
    *,
    wall_s: float | None = None,
    ssd_read_ops: int | None = None,
    page_size: int | None = None,
) -> None:
    """
    Print GIDS / loader feature timing and effective feature GB/s (decimal GB = 1e9 bytes).

    If ``wall_s`` is set, prints two lines: times (e2e, feature aggregation, remainder as
    model training), then counts and bandwidths. Optional ``ssd_read_ops`` and ``page_size``
    add ``ssd_bandwidth = (ssd_read_ops * page_size) / feat_agg_s`` in GB/s.
    """
    if n_feat_nodes <= 0 or feat_agg_s <= 0:
        print(f"{prefix}BAM feature fetch: no samples or zero time.")
        return
    bytes_moved = float(n_feat_nodes * feat_dim * 4)
    feat_gbps = (bytes_moved / 1e9) / feat_agg_s

    if wall_s is not None:
        wall_f = float(wall_s)
        model_s = max(0.0, wall_f - float(feat_agg_s))
        print(
            f"{prefix}BAM times: end_to_end={wall_f:.4f}s  "
            f"feature_aggregation={float(feat_agg_s):.4f}s  "
            f"model_training={model_s:.4f}s"
        )
        line2 = (
            f"{prefix}BAM data: feature_nodes={n_feat_nodes}  feat_dim={feat_dim}  "
            f"effective_bandwidth={feat_gbps:.3f} GB/s"
        )
        if ssd_read_ops is not None and page_size is not None:
            ssd_bytes = float(ssd_read_ops) * float(page_size)
            ssd_gbps = (ssd_bytes / 1e9) / float(feat_agg_s)
            line2 += (
                f"  ssd_read_ops={int(ssd_read_ops)}  "
                f"ssd_bandwidth={ssd_gbps:.3f} GB/s"
            )
        print(line2)
        return

    print(
        f"{prefix}BAM feature aggregation: time={feat_agg_s:.4f}s  "
        f"feature_nodes={n_feat_nodes}  feat_dim={feat_dim}  "
        f"effective_bandwidth={feat_gbps:.3f} GB/s"
    )


def train_epoch(
    loader,
    model,
    optimizer,
    device,
    use_bam: bool,
    gids,
    feat_dim: int,
    *,
    bam_features_from_loader: bool = False,
    max_batches: int | None = None,
):
    """Returns (mean_loss, feat_fetch_seconds, n_feature_nodes, n_batches) for BAM timing.

    If ``bam_features_from_loader`` is True (e.g. ``LSM_GNN_Neighbor_Loader`` already
    filled ``batch.x`` via ``GIDS``), skip ``fetch_feature`` here; aggregate fetch
    timing should be recorded in the loader's ``feature_fn`` instead.
    """
    model.train()
    total_loss = 0.0
    n = 0
    feat_fetch_s = 0.0
    n_feature_nodes = 0
    n_batches = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_bam:
            if not hasattr(batch, "n_id") or batch.n_id is None:
                raise RuntimeError(
                    "Batch missing n_id; use a recent PyG NeighborLoader for BAM gathers."
                )
            if bam_features_from_loader:
                x = batch.x
                if x is None or x.dim() != 2 or x.size(1) != feat_dim:
                    raise RuntimeError(
                        "Expected batch.x from loader feature_fn with shape "
                        f"[..., {feat_dim}], got {None if x is None else tuple(x.shape)}"
                    )
            else:
                idx = batch.n_id.to(device, dtype=torch.long)
                t0 = time.perf_counter()
                x = gids.fetch_feature(idx, feat_dim)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                feat_fetch_s += time.perf_counter() - t0
                n_feature_nodes += int(idx.numel())
        else:
            x = batch.x
        logits = model(x, batch.edge_index)[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.batch_size
        n += batch.batch_size
        n_batches = i + 1
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return total_loss / max(n, 1), feat_fetch_s, n_feature_nodes, n_batches


@torch.no_grad()
def eval_epoch(
    loader,
    model,
    device,
    use_bam: bool,
    gids,
    feat_dim: int,
    *,
    bam_features_from_loader: bool = False,
    max_batches: int | None = None,
):
    """Returns (accuracy_percent, num_batches)."""
    model.eval()
    correct = 0
    tot = 0
    n_batches = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        if use_bam:
            if bam_features_from_loader:
                x = batch.x
                if x is None or x.dim() != 2 or x.size(1) != feat_dim:
                    raise RuntimeError(
                        "Expected batch.x from loader feature_fn with shape "
                        f"[..., {feat_dim}], got {None if x is None else tuple(x.shape)}"
                    )
            else:
                idx = batch.n_id.to(device, dtype=torch.long)
                x = gids.fetch_feature(idx, feat_dim)
        else:
            x = batch.x
        logits = model(x, batch.edge_index)[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum())
        tot += int(y.numel())
        n_batches = i + 1
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return 100.0 * correct / max(tot, 1), n_batches


def train_epoch_fullbatch(data, model, optimizer, device, use_bam: bool, gids, feat_dim: int):
    """Full-graph forward; no pyg-lib / torch-sparse required.
    Returns (loss, feat_fetch_seconds, n_feature_nodes)."""
    model.train()
    optimizer.zero_grad()
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    feat_fetch_s = 0.0
    n_feature_nodes = 0
    if use_bam:
        idx = torch.arange(data.num_nodes, device=device, dtype=torch.long)
        t0 = time.perf_counter()
        x = gids.fetch_feature(idx, feat_dim)
        if device.type == "cuda":
            torch.cuda.synchronize()
        feat_fetch_s = time.perf_counter() - t0
        n_feature_nodes = int(idx.numel())
    else:
        x = data.x.to(device)
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach()), feat_fetch_s, n_feature_nodes


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
    if args.bam:
        data = data_without_dense_node_features(data)
        print(
            f"Slim BAM Data: x shape {tuple(data.x.shape)} (GIDS-only features; "
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
    print(
        f"Test acc {test_acc:.2f}% | test_batches={n_te} | wall {time.time() - t0:.1f}s"
    )
    if use_bam:
        print_bam_feature_fetch_summary(
            total_feat_fetch_s, total_n_feature_nodes, args.feat_dim
        )


if __name__ == "__main__":
    main()
