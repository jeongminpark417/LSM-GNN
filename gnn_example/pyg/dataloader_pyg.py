"""
PyTorch Geometric loaders mirroring ``gnn_example/dataloader.py`` (IGB / OGB).

Uses the same paths, mmap flags, and masks as the DGL datasets, but builds
``torch_geometric.data.Data`` or ``HeteroData`` (no DGL dependency).

**Full IGB (homogeneous):** DGL uses on-disk CSC; PyG needs COO ``edge_index``.
This module converts CSC npy files (same paths as ``IGB260MDGLDataset``) to
``edge_index`` — this can require **enormous** RAM for the full graph.

Example::

    from dataloader_pyg import build_homogeneous_pyg_data, SimpleIGBArgs
    args = SimpleIGBArgs(path="/data/IGB", dataset_size="small", data="IGB")
    data = build_homogeneous_pyg_data(args)
"""

from __future__ import annotations

import argparse
import os.path as osp
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

# -----------------------------------------------------------------------------
# IGB260M — duplicated from gnn_example/dataloader.py (no dgl import).
# Keep in sync when paths or layout change.
# -----------------------------------------------------------------------------


class IGB260M(object):
    def __init__(
        self,
        root: str,
        size: str,
        in_memory: int,
        uva_graph: int,
        classes: int,
        synthetic: int,
        emb_size: int,
        data: str,
    ):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes
        self.emb_size = emb_size
        self.uva_graph = uva_graph
        self.data = data

    def num_nodes(self):
        if self.data == "OGB":
            return 111059956

        if self.size == "experimental":
            return 100000
        elif self.size == "small":
            return 1000000
        elif self.size == "medium":
            return 10000000
        elif self.size == "large":
            return 100000000
        elif self.size == "full":
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        if self.data == "OGB":
            path = osp.join(self.dir, "node_feat.npy")
            if self.in_memory:
                emb = np.load(path)
            else:
                emb = np.load(path, mmap_mode="r")

        elif self.size == "large" or self.size == "full":
            path = "/mnt/nvme16/node_feat.npy"
            if self.in_memory:
                emb = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes, 1024)
                ).copy()
            else:
                emb = np.memmap(
                    path, dtype="float32", mode="r", shape=(num_nodes, 1024)
                )
        else:
            path = osp.join(
                self.dir, self.size, "processed", "paper", "node_feat.npy"
            )
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype("f")
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode="r")

        return emb

    @property
    def paper_label(self) -> np.ndarray:
        if self.data == "OGB":
            return np.random.randint(low=0, size=111059956, high=171)
        elif self.size == "large" or self.size == "full":
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = "/mnt/nvme15/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy"
                if self.in_memory:
                    node_labels = np.memmap(
                        path, dtype="float32", mode="r", shape=(num_nodes,)
                    ).copy()
                else:
                    node_labels = np.memmap(
                        path, dtype="float32", mode="r", shape=(num_nodes,)
                    )
            else:
                path = "/mnt/nvme15/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy"

                if self.in_memory:
                    node_labels = np.load(path)
                else:
                    node_labels = np.memmap(
                        path, dtype="float32", mode="r", shape=(num_nodes,)
                    )

        else:
            if self.num_classes == 19:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_19.npy"
                )
            else:
                path = osp.join(
                    self.dir, self.size, "processed", "paper", "node_label_2K.npy"
                )
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode="r")
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(
            self.dir,
            self.size,
            "processed",
            "paper__cites__paper",
            "edge_index.npy",
        )
        if self.data == "OGB":
            path = osp.join(self.dir, "edge_index.npy")
        elif self.size == "full":
            path = "/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy"
        elif self.size == "large":
            path = "/mnt/nvme7/large/processed/paper__cites__paper/edge_index.npy"

        if self.in_memory or self.uva_graph:
            return np.load(path)
        else:
            return np.load(path, mmap_mode="r")


def csc_npy_to_edge_index(
    edge_col_idx: torch.Tensor,
    edge_row_idx: torch.Tensor,
    _edge_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert DGL-style CSC triple (col_ptr, row_ind, edge_id) to PyG COO [2, E].
    ``edge_col_idx`` = column pointers (length N+1), ``edge_row_idx`` = row indices.
    """
    ptr = edge_col_idx.long()
    row = edge_row_idx.long()
    if ptr.numel() < 2:
        return torch.empty(2, 0, dtype=torch.long)
    deg = ptr[1:] - ptr[:-1]
    num_cols = deg.numel()
    col = torch.repeat_interleave(
        torch.arange(num_cols, device=ptr.device, dtype=torch.long), deg
    )
    return torch.stack([row, col], dim=0)


def _homogeneous_masks(
    dataset_size: str, num_classes: int, n_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset_size == "full":
        if num_classes == 19:
            n_labeled_idx = 227130858
        else:
            n_labeled_idx = 157675969
        n_train = int(n_labeled_idx * 0.6)
        n_val = int(n_labeled_idx * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val : n_labeled_idx] = True
    else:
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
    return train_mask, val_mask, test_mask


@dataclass
class SimpleIGBArgs:
    """Minimal namespace compatible with ``gids_training.py`` argparse fields."""

    path: str = "/mnt/nvme14/IGB260M"
    dataset_size: str = "experimental"
    in_memory: int = 0
    uva_graph: int = 0
    num_classes: int = 19
    synthetic: int = 0
    emb_size: int = 1024
    data: str = "IGB"


def build_homogeneous_pyg_data(args: Any) -> "torch_geometric.data.Data":
    """IGB or OGB homogeneous graph as PyG ``Data`` (mirrors IGB260MDGLDataset / OGBDGLDataset)."""
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, remove_self_loops

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

    x = torch.from_numpy(dataset.paper_feat).float()
    node_labels = torch.from_numpy(dataset.paper_label).long()
    node_edges = torch.from_numpy(dataset.paper_edge)

    if getattr(args, "data", "IGB") == "OGB":
        edge_index = torch.stack([node_edges[0, :], node_edges[1, :]], dim=0).long()
    elif args.dataset_size == "full":
        csc_row = torch.from_numpy(
            np.load(
                "/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_row_idx.npy"
            )
        )
        csc_col = torch.from_numpy(
            np.load(
                "/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_col_idx.npy"
            )
        )
        csc_e = torch.from_numpy(
            np.load(
                "/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index_csc_edge_idx.npy"
            )
        )
        edge_index = csc_npy_to_edge_index(csc_col, csc_row, csc_e)
    else:
        edge_index = torch.stack([node_edges[:, 0], node_edges[:, 1]], dim=0).long()

    if args.dataset_size != "full":
        edge_index, _ = remove_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    train_mask, val_mask, test_mask = _homogeneous_masks(
        args.dataset_size, args.num_classes, x.size(0)
    )

    data = Data(x=x, edge_index=edge_index, y=node_labels, num_nodes=x.size(0))
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def build_hetero_igb_pyg(args: Any) -> "torch_geometric.data.HeteroData":
    """IGB heterogeneous (small layout) -> ``HeteroData`` (mirrors IGBHeteroDGLDataset)."""
    from torch_geometric.data import HeteroData

    d = args.path
    size = args.dataset_size
    H = HeteroData()

    def _load_edges(rel_path: str, mmap: bool):
        p = osp.join(d, size, "processed", rel_path)
        arr = np.load(p) if not mmap else np.load(p, mmap_mode="r")
        t = torch.from_numpy(np.asarray(arr))
        return t[:, 0].long(), t[:, 1].long()

    mmap = not args.in_memory
    p0, p1 = _load_edges("paper__cites__paper/edge_index.npy", mmap)
    H["paper", "cites", "paper"].edge_index = torch.stack([p0, p1], dim=0)
    p0, p1 = _load_edges("paper__written_by__author/edge_index.npy", mmap)
    H["paper", "written_by", "author"].edge_index = torch.stack([p0, p1], dim=0)
    p0, p1 = _load_edges("author__affiliated_to__institute/edge_index.npy", mmap)
    H["author", "affiliated_to", "institute"].edge_index = torch.stack([p0, p1], dim=0)
    p0, p1 = _load_edges("paper__topic__fos/edge_index.npy", mmap)
    H["paper", "topic", "fos"].edge_index = torch.stack([p0, p1], dim=0)

    def _nf(rel: str, mmap: bool):
        p = osp.join(d, size, "processed", rel)
        a = np.load(p) if not mmap else np.load(p, mmap_mode="r")
        return torch.from_numpy(np.asarray(a)).float()

    H["paper"].x = _nf("paper/node_feat.npy", mmap)
    H["paper"].y = (
        torch.from_numpy(
            np.asarray(
                np.load(osp.join(d, size, "processed", "paper/node_label_19.npy"))
                if not mmap
                else np.load(
                    osp.join(d, size, "processed", "paper/node_label_19.npy"),
                    mmap_mode="r",
                )
            )
        )
        .long()
    )
    H["author"].x = _nf("author/node_feat.npy", mmap)
    H["institute"].x = _nf("institute/node_feat.npy", mmap)
    H["fos"].x = _nf("fos/node_feat.npy", mmap)

    n_paper = H["paper"].x.size(0)
    n_train = int(n_paper * 0.6)
    n_val = int(n_paper * 0.2)
    tr = torch.zeros(n_paper, dtype=torch.bool)
    va = torch.zeros(n_paper, dtype=torch.bool)
    te = torch.zeros(n_paper, dtype=torch.bool)
    tr[:n_train] = True
    va[n_train : n_train + n_val] = True
    te[n_train + n_val :] = True
    H["paper"].train_mask = tr
    H["paper"].val_mask = va
    H["paper"].test_mask = te
    return H


def build_hetero_igb_massive_pyg(args: Any) -> "torch_geometric.data.HeteroData":
    """IGB ``full`` / ``large`` hetero (mirrors IGBHeteroDGLDatasetMassive)."""
    from torch_geometric.data import HeteroData

    d = args.path
    size = args.dataset_size
    mmap = not args.uva_graph and not args.in_memory

    def _e(rel: str):
        p = osp.join(d, size, "processed", rel)
        a = np.load(p) if not mmap else np.load(p, mmap_mode="r")
        t = torch.from_numpy(np.asarray(a))
        return torch.stack([t[:, 0].long(), t[:, 1].long()], dim=0)

    H = HeteroData()
    H["paper", "cites", "paper"].edge_index = _e("paper__cites__paper/edge_index.npy")
    H["paper", "written_by", "author"].edge_index = _e(
        "paper__written_by__author/edge_index.npy"
    )
    H["author", "affiliated_to", "institute"].edge_index = _e(
        "author__affiliated_to__institute/edge_index.npy"
    )
    H["paper", "topic", "fos"].edge_index = _e("paper__topic__fos/edge_index.npy")

    if size == "full":
        num_paper_nodes = 269346174
        pfeat = np.memmap(
            osp.join(d, "full", "processed", "paper", "node_feat.npy"),
            dtype="float32",
            mode="r",
            shape=(num_paper_nodes, 1024),
        )
        if args.num_classes == 19:
            plab = np.memmap(
                osp.join(d, "full", "processed", "paper", "node_label_19.npy"),
                dtype="float32",
                mode="r",
                shape=(num_paper_nodes,),
            )
        else:
            plab = np.memmap(
                osp.join(d, "full", "processed", "paper", "node_label_2K.npy"),
                dtype="float32",
                mode="r",
                shape=(num_paper_nodes,),
            )
        num_author_nodes = 277220883
        author_node_features = np.memmap(
            osp.join(d, "full", "processed", "author", "node_feat.npy"),
            dtype="float32",
            mode="r",
            shape=(num_author_nodes, 1024),
        )
    elif size == "large":
        num_paper_nodes = 100000000
        pfeat = np.memmap(
            osp.join(d, "full", "processed", "paper", "node_feat.npy"),
            dtype="float32",
            mode="r",
            shape=(num_paper_nodes, 1024),
        )
        if args.num_classes == 19:
            plab = np.memmap(
                osp.join(d, "full", "processed", "paper", "node_label_19.npy"),
                dtype="float32",
                mode="r",
                shape=(num_paper_nodes,),
            )
        else:
            plab = np.memmap(
                osp.join(d, "full", "processed", "paper", "node_label_2K.npy"),
                dtype="float32",
                mode="r",
                shape=(num_paper_nodes,),
            )
        num_author_nodes = 116959896
        author_node_features = np.memmap(
            osp.join(d, "full", "processed", "author", "node_feat.npy"),
            dtype="float32",
            mode="r",
            shape=(num_author_nodes, 1024),
        )
    else:
        raise ValueError("build_hetero_igb_massive_pyg expects dataset_size full or large")

    institute_node_features = np.load(
        osp.join(d, size, "processed", "institute", "node_feat.npy"),
        mmap_mode="r" if mmap else None,
    )
    fos_node_features = np.load(
        osp.join(d, size, "processed", "fos", "node_feat.npy"),
        mmap_mode="r" if mmap else None,
    )

    H["paper"].x = torch.from_numpy(np.asarray(pfeat))
    H["paper"].y = torch.from_numpy(np.asarray(plab)).long()
    H["author"].x = torch.from_numpy(np.asarray(author_node_features))
    H["institute"].x = torch.from_numpy(np.asarray(institute_node_features)).float()
    H["fos"].x = torch.from_numpy(np.asarray(fos_node_features)).float()

    n_paper = num_paper_nodes
    n_train = int(n_paper * 0.6)
    n_val = int(n_paper * 0.2)
    tr = torch.zeros(n_paper, dtype=torch.bool)
    va = torch.zeros(n_paper, dtype=torch.bool)
    te = torch.zeros(n_paper, dtype=torch.bool)
    tr[:n_train] = True
    va[n_train : n_train + n_val] = True
    te[n_train + n_val :] = True
    H["paper"].train_mask = tr
    H["paper"].val_mask = va
    H["paper"].test_mask = te
    return H


def build_hetero_ogb_massive_pyg(args: Any) -> "torch_geometric.data.HeteroData":
    """OGB MAG-style hetero (mirrors OGBHeteroDGLDatasetMassive)."""
    from torch_geometric.data import HeteroData

    d = args.path
    mmap = not (args.uva_graph or args.in_memory)

    def _e(rel: str):
        p = osp.join(d, "processed", rel)
        a = np.load(p) if not mmap else np.load(p, mmap_mode="r")
        t = torch.from_numpy(np.asarray(a))
        return torch.stack([t[0, :].long(), t[1, :].long()], dim=0)

    H = HeteroData()
    H["paper", "cites", "paper"].edge_index = _e("paper___cites___paper/edge_index.npy")
    H["author", "writes", "paper"].edge_index = _e(
        "author___writes___paper/edge_index.npy"
    )
    H["author", "affiliated_to", "institute"].edge_index = _e(
        "author___affiliated_with___institution/edge_index.npy"
    )

    num_paper_nodes = 121751666
    paper_feat = np.load(
        osp.join(d, "processed", "paper", "node_feat.npy"), mmap_mode="r" if mmap else None
    )
    paper_lab = np.load(
        osp.join(d, "processed", "paper", "node_label.npy"), mmap_mode="r" if mmap else None
    )
    pl = torch.from_numpy(np.asarray(paper_lab)).long()
    pl[pl < 0] = 0

    H["paper"].x = torch.from_numpy(np.asarray(paper_feat)).to(torch.float32)
    H["paper"].y = pl

    n_nodes = H["paper"].x.size(0)
    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)
    tr = torch.zeros(n_nodes, dtype=torch.bool)
    va = torch.zeros(n_nodes, dtype=torch.bool)
    te = torch.zeros(n_nodes, dtype=torch.bool)
    tr[:n_train] = True
    va[n_train : n_train + n_val] = True
    te[n_train + n_val :] = True
    H["paper"].train_mask = tr
    H["paper"].val_mask = va
    H["paper"].test_mask = te
    return H


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG Data / HeteroData (IGB/OGB)")
    parser.add_argument("--path", type=str, default="/mnt/nvme15/IGB260M_part_2")
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="full",
        choices=["experimental", "small", "medium", "large", "full"],
    )
    parser.add_argument("--num_classes", type=int, default=2983, choices=[19, 2983])
    parser.add_argument("--in_memory", type=int, default=0, choices=[0, 1])
    parser.add_argument("--uva_graph", type=int, default=0, choices=[0, 1])
    parser.add_argument("--synthetic", type=int, default=0, choices=[0, 1])
    parser.add_argument("--emb_size", type=int, default=1024)
    parser.add_argument("--data", type=str, default="IGB", choices=["IGB", "OGB"])
    parser.add_argument(
        "--kind",
        type=str,
        default="homo",
        choices=["homo", "hetero", "hetero_massive", "ogb_hetero_massive"],
    )
    ns = parser.parse_args()

    t0 = time.time()
    if ns.kind == "homo":
        out = build_homogeneous_pyg_data(ns)
        print(out)
        print("edge_index", out.edge_index.shape, "x", out.x.shape)
    elif ns.kind == "hetero":
        out = build_hetero_igb_pyg(ns)
        print(out)
    elif ns.kind == "hetero_massive":
        out = build_hetero_igb_massive_pyg(ns)
        print(out)
    else:
        out = build_hetero_ogb_massive_pyg(ns)
        print(out)
    print("elapsed_s", time.time() - t0)
