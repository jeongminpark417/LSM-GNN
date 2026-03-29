#!/usr/bin/env python3
"""
Micro-benchmark and smoke tests for ``LSM_NVMeFeatureClient.update_prefetch_timestamp``.

**CUDA is required** (tensors must live on GPU).

Examples::

    # Python API + dtype/shape checks only. Skips ``init_controllers``; the C++
    # path returns immediately when the backing array is not constructed (no NVMe).
    python bench_update_prefetch_timestamp.py --skip-nvme-init

    # Full path: needs BAM/libnvm on ``LD_LIBRARY_PATH`` and NVMe devices.
    export LD_LIBRARY_PATH=/path/to/LSM-GNN/bam/build/lib:$LD_LIBRARY_PATH
    python bench_update_prefetch_timestamp.py --lsm-build /path/to/lsm_module/build \\
        --repeats 100 --warmup 10

    # Check packed next_reuse vs expected (timestamp << 48) | idx (needs NVMe init).
    python bench_update_prefetch_timestamp.py --lsm-build .../lsm_module/build --verify

Repo layout assumes this file lives under ``gnn_example/pyg/lsm_nvme_pyg/micro_bench/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_LSM_NVME_PYG = os.path.normpath(os.path.join(_HERE, ".."))
if _LSM_NVME_PYG not in sys.path:
    sys.path.insert(0, _LSM_NVME_PYG)

import torch

from lsm_nvme_client import (
    LSM_NVMeFeatureClient,
    NEXT_REUSE_NOT_RESIDENT_U64,
    as_u64_from_signed_i64,
    pack_prefetch_timestamp_idx,
    unpack_prefetch_idx,
    unpack_prefetch_timestamp,
)


def _assert_raises(fn, exc_type: type[BaseException]) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def run_validation_tests(device: torch.device) -> None:
    """Client-side checks (no NVMe init)."""
    client = LSM_NVMeFeatureClient(no_init=True, ctrl_idx=device.index or 0)

    n = 16
    pages = torch.randint(0, 1024, (n,), dtype=torch.int64, device=device)
    ts = torch.randint(0, 2**16, (n,), dtype=torch.int32, device=device)
    idxs = torch.arange(n, dtype=torch.int64, device=device)

    _assert_raises(
        lambda: client.update_prefetch_timestamp(
            pages.cpu(), ts, idxs,
        ),
        ValueError,
    )
    _assert_raises(
        lambda: client.update_prefetch_timestamp(
            pages, ts[: n // 2], idxs,
        ),
        ValueError,
    )

    client.update_prefetch_timestamp(pages, ts, idxs)
    if device.type == "cuda":
        torch.cuda.synchronize()


def check_next_reuse_matches_expected(
    raw_i64: torch.Tensor,
    expected_packed_u64: int,
    *,
    context: str,
) -> None:
    """Decode device readback and compare to expected packed (ts<<48)|idx."""
    if raw_i64.numel() != 1:
        raise ValueError("check_next_reuse_matches_expected expects a single element")
    got = as_u64_from_signed_i64(int(raw_i64.item()))
    exp = int(expected_packed_u64) % (1 << 64)
    if got == NEXT_REUSE_NOT_RESIDENT_U64:
        raise AssertionError(f"{context}: page not resident (readback ~0)")
    if got != exp:
        raise AssertionError(
            f"{context}: next_reuse={got:#x} ({unpack_prefetch_timestamp(got)=}, "
            f"{unpack_prefetch_idx(got)=}) != expected {exp:#x}"
        )


def run_verify_correctness(
    client: LSM_NVMeFeatureClient,
    device: torch.device,
    *,
    page_size: int,
    feat_dim: int,
) -> None:
    """
    Bring logical page 0 into cache via ``fetch_feature``, write one prefetch
    record, read back ``next_reuse`` and check it matches ``pack_prefetch_timestamp_idx``.

    Then apply a second ``atomicMin`` with a smaller packed value and assert the
    minimum wins.
    """
    elems_per_page = page_size // 4  # float32
    row0 = torch.tensor([0], device=device, dtype=torch.long)
    _ = client.fetch_feature(row0, feat_dim)
    torch.cuda.synchronize()

    logical = torch.zeros(1, dtype=torch.int64, device=device)
    rb0 = client.read_next_reuse_for_pages(logical)
    u0 = as_u64_from_signed_i64(int(rb0.item()))
    if u0 == NEXT_REUSE_NOT_RESIDENT_U64:
        raise AssertionError(
            "verify: logical page 0 not resident after fetch_feature; "
            "cannot test next_reuse."
        )

    ts_a = 0x1234ABCD
    idx_a = 0x0000_EEEE_DDDD_CCCC & ((1 << 48) - 1)
    client.update_prefetch_timestamp(
        logical,
        torch.tensor([ts_a], dtype=torch.int32, device=device),
        torch.tensor([idx_a], dtype=torch.int64, device=device),
    )
    torch.cuda.synchronize()

    exp_a = pack_prefetch_timestamp_idx(ts_a, idx_a)
    check_next_reuse_matches_expected(
        client.read_next_reuse_for_pages(logical),
        exp_a,
        context="single update",
    )
    print(
        f"  verify single update: next_reuse={exp_a:#x}  "
        f"ts={unpack_prefetch_timestamp(exp_a)}  idx={unpack_prefetch_idx(exp_a)}",
        flush=True,
    )

    ts_b, idx_b = 5, 0x200
    ts_c, idx_c = 10, 0x100
    packed_b = pack_prefetch_timestamp_idx(ts_b, idx_b)
    packed_c = pack_prefetch_timestamp_idx(ts_c, idx_c)
    exp_min = packed_b if packed_b < packed_c else packed_c

    client.update_prefetch_timestamp(
        logical,
        torch.tensor([ts_c], dtype=torch.int32, device=device),
        torch.tensor([idx_c], dtype=torch.int64, device=device),
    )
    client.update_prefetch_timestamp(
        logical,
        torch.tensor([ts_b], dtype=torch.int32, device=device),
        torch.tensor([idx_b], dtype=torch.int64, device=device),
    )
    torch.cuda.synchronize()

    check_next_reuse_matches_expected(
        client.read_next_reuse_for_pages(logical),
        exp_min,
        context="atomicMin of two updates",
    )
    print(
        f"  verify atomicMin: expected_min={exp_min:#x} "
        f"(ts={unpack_prefetch_timestamp(exp_min)}, idx={unpack_prefetch_idx(exp_min)})",
        flush=True,
    )

    _ = elems_per_page  # reserved if we extend to multi-page tests


def run_benchmark(
    client: LSM_NVMeFeatureClient,
    device: torch.device,
    n: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> None:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    pages = torch.randint(0, 1 << 20, (n,), dtype=torch.int64, device=device, generator=g)
    ts = torch.randint(0, 2**31, (n,), dtype=torch.int32, device=device, generator=g)
    idxs = torch.randint(0, 1 << 48, (n,), dtype=torch.int64, device=device, generator=g)

    for _ in range(warmup):
        client.update_prefetch_timestamp(pages, ts, idxs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            client.update_prefetch_timestamp(pages, ts, idxs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        total_s = ms / 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            client.update_prefetch_timestamp(pages, ts, idxs)
        total_s = time.perf_counter() - t0

    per_us = (total_s / repeats) * 1e6
    print(
        f"update_prefetch_timestamp: n={n}  repeats={repeats}  "
        f"total={total_s*1e3:.3f} ms  per_call={per_us:.2f} µs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test / micro-bench LSM_NVMe update_prefetch_timestamp",
    )
    parser.add_argument(
        "--skip-nvme-init",
        action="store_true",
        help="Use LSM_NVMeFeatureClient(no_init=True); kernel is skipped in C++.",
    )
    parser.add_argument(
        "--lsm-build",
        type=str,
        default=None,
        help="lsm_module CMake build dir (contains LSM_NVMe/).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="LSM-GNN repo root (default: inferred from this file).",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=4096, help="Elements per call.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="Only run validation tests, no timing loop.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After benchmark (or with --verify-only), check readback next_reuse vs packed ts|idx.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run correctness check only (implies full NVMe init; conflicts with --skip-nvme-init).",
    )
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--feat-dim", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA is required for update_prefetch_timestamp.", file=sys.stderr)
        sys.exit(1)

    if args.verify_only and args.skip_nvme_init:
        print("error: --verify-only requires a real store (omit --skip-nvme-init).", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    print("Running validation tests...", flush=True)
    run_validation_tests(device)
    print("  ok", flush=True)

    if args.no_bench and not args.verify and not args.verify_only:
        return

    client = LSM_NVMeFeatureClient(
        no_init=args.skip_nvme_init and not args.verify_only,
        ctrl_idx=args.device,
        lsm_build=args.lsm_build,
        repo_root=args.repo_root,
        cache_size=1,
        num_ele=1024,
        cache_dim=max(1, args.feat_dim),
        page_size=args.page_size,
    )
    mode = "skip_nvme_init (C++ no-op)" if args.skip_nvme_init and not args.verify_only else "full init"
    print(f"Benchmark mode: {mode}", flush=True)

    if args.verify_only or args.verify:
        if args.skip_nvme_init and not args.verify_only:
            print("error: --verify requires full init (omit --skip-nvme-init).", file=sys.stderr)
            sys.exit(1)
        print("Running next_reuse correctness check...", flush=True)
        run_verify_correctness(
            client,
            device,
            page_size=args.page_size,
            feat_dim=max(1, args.feat_dim),
        )
        print("  verify ok", flush=True)

    if args.verify_only:
        return

    if args.no_bench:
        return

    run_benchmark(
        client,
        device,
        n=max(1, args.n),
        warmup=max(0, args.warmup),
        repeats=max(1, args.repeats),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
