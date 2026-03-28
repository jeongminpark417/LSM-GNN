#!/usr/bin/env python3
"""
Minimal example for the LSM-GNN BAM_Feature_Store extension.

Build first (from repo root or lsm_module):

  cmake -S lsm_module -B lsm_module/build -DLSM_CUDA_ARCHITECTURES=80
  cmake --build lsm_module/build -j

Run this script with the CMake build directory on PYTHONPATH so Python finds
the package under ``<build>/BAM_Feature_Store/`` (contains ``__init__.py`` and
the ``.so``).

If import fails with "error while loading shared libraries: libnvm.so", add
BAM's library directory, for example:

  export LD_LIBRARY_PATH="/path/to/LSM-GNN/bam/build/lib:${LD_LIBRARY_PATH}"

``init_controllers`` matches the C++ binding (see gids_nvme.cu):

  init_controllers(ps, read_off, cache_size_gb, num_ele, num_ssd, wb_size,
                   wb_queue_size, cpu_agg, cpu_agg_q_depth)

- ps: page size in bytes (e.g. 4096; feature dim = ps / sizeof(float))
- read_off: NVMe read offset
- cache_size_gb: host-side cache size in **gibibytes** (GiB)
- num_ele: number of feature elements in backing store
- num_ssd: number of BAM controllers / SSDs
- wb_size, wb_queue_size: window-buffer depth and queue length
- cpu_agg, cpu_agg_q_depth: CPU aggregation path
"""

from __future__ import annotations

import argparse
import os
import sys


def _prepend_sys_path(build_dir: str) -> None:
    b = os.path.abspath(build_dir)
    if b not in sys.path:
        sys.path.insert(0, b)


def main() -> None:
    p = argparse.ArgumentParser(description="BAM_Feature_Store smoke test / example")
    p.add_argument(
        "--build-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "build"),
        help="lsm_module CMake build directory (parent of BAM_Feature_Store/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only import BAM_Feature_Store; do not open NVMe or call init_controllers",
    )
    p.add_argument("--ps", type=int, default=4096, help="page size (bytes)")
    p.add_argument("--read-off", type=int, default=0, help="read offset")
    p.add_argument("--cache-gb", type=int, default=1, help="cache size (GiB)")
    p.add_argument("--num-ele", type=int, default=100, help="num elements (backing store)")
    p.add_argument("--num-ssd", type=int, default=1, help="number of controllers")
    p.add_argument("--wb-size", type=int, default=4, help="window buffer depth")
    p.add_argument("--wb-queue", type=int, default=131072, help="WB queue size")
    p.add_argument("--cpu-agg", action="store_true", help="enable CPU aggregation")
    p.add_argument("--cpu-agg-q-depth", type=int, default=0)
    p.add_argument(
        "--demo-read-feature",
        action="store_true",
        help="After init, run a tiny read_feature with PyTorch (requires torch + CUDA)",
    )
    args = p.parse_args()

    _prepend_sys_path(args.build_dir)

    import BAM_Feature_Store  # noqa: E402

    print("Imported BAM_Feature_Store from:", BAM_Feature_Store.__file__)

    if args.dry_run:
        print("Dry run OK.")
        return

    store = BAM_Feature_Store.BAM_Feature_Store()
    store.init_controllers(
        args.ps,
        args.read_off,
        args.cache_gb,
        args.num_ele,
        args.num_ssd,
        args.wb_size,
        args.wb_queue,
        args.cpu_agg,
        args.cpu_agg_q_depth,
    )
    print("init_controllers finished.")

    if args.demo_read_feature:
        import torch

        dim = args.ps // 4
        cache_dim = dim
        n = 4
        out = torch.zeros(n, dim, device="cuda", dtype=torch.float32)
        idx = torch.arange(n, device="cuda", dtype=torch.int64)
        store.read_feature(
            out.data_ptr(),
            idx.data_ptr(),
            n,
            dim,
            cache_dim,
        )
        torch.cuda.synchronize()
        print("read_feature demo OK, out shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
