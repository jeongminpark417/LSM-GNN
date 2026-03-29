#!/usr/bin/env python3
"""
Write a raw ``.bin`` file to the BaM NVMe backing store used by GIDS.

``BAM_Feature_Store`` / ``GIDS`` do **not** expose a Python ``write_feature``;
loads use ``read_feature`` only. The supported way to populate the device is
``bam/build/bin/nvm-readwrite-bench`` (``--access_type=1``), which issues NVMe
writes from the GPU and **already** waits with ``cudaDeviceSynchronize()`` per
chunk before exit.

This script runs that benchmark with your paths, then optionally:

- calls ``torch.cuda.synchronize()`` on the visible CUDA device (quiet GPU work);
- runs the host ``sync(8)`` helper (best-effort; does **not** issue NVMe Flush
  on the libnvm path, but is harmless if you want a full block-layer flush).

Usage (root typically required for ``/dev/libnvm*``)::

    sudo env CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/bpark/bam/build/lib:\\$LD_LIBRARY_PATH \\
      /home/bpark/miniconda3/bin/python3 gids_backing_write_from_bin.py \\
      --input /home/bpark/sequential_floats.bin \\
      --gpu 0

``CUDA_VISIBLE_DEVICES=2`` + ``--gpu 0`` → physical GPU 2, same convention as
``verify_gids_igb_embeddings.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _sync_host() -> None:
    try:
        os.sync()
    except AttributeError:
        subprocess.run(["sync"], check=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write .bin to NVMe via nvm-readwrite-bench (GIDS backing store)"
    )
    p.add_argument("--input", type=str, required=True, help="Raw file to write")
    p.add_argument(
        "--readwrite-bench",
        type=str,
        default="/home/bpark/bam/build/bin/nvm-readwrite-bench",
        help="Path to nvm-readwrite-bench",
    )
    p.add_argument(
        "--bam-lib",
        type=str,
        default="/home/bpark/bam/build/lib",
        help="Prepended to LD_LIBRARY_PATH for libnvm.so",
    )
    p.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="If set, exported as CUDA_VISIBLE_DEVICES for the child process",
    )
    p.add_argument("--gpu", type=int, default=0, help="Bench --gpu (index in visible set)")
    p.add_argument("--n_ctrls", type=int, default=1)
    p.add_argument("--threads", type=int, default=1024)
    p.add_argument("--blk_size", type=int, default=64)
    p.add_argument("--pages", type=int, default=1024)
    p.add_argument("--page_size", type=int, default=4096)
    p.add_argument("--queue_depth", type=int, default=1024)
    p.add_argument("--num_queues", type=int, default=128)
    p.add_argument("--num_blks", type=int, default=2097152)
    p.add_argument("--random", type=str, default="false", choices=["true", "false"])
    p.add_argument("--reqs", type=int, default=1)
    p.add_argument("--loffset", type=int, default=0, help="NVMe byte offset (bench --loffset)")
    p.add_argument("--ioffset", type=int, default=0, help="Input file byte skip (bench --ioffset)")
    p.add_argument(
        "--skip-torch-sync",
        action="store_true",
        help="Do not call torch.cuda.synchronize() after bench",
    )
    p.add_argument(
        "--skip-host-sync",
        action="store_true",
        help="Do not call os.sync() / sync after bench",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only, do not run",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.readwrite_bench):
        print(f"error: nvm-readwrite-bench not found: {args.readwrite_bench}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    bam_lib = os.path.abspath(args.bam_lib)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = bam_lib + (os.pathsep + old_ld if old_ld else "")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        args.readwrite_bench,
        f"--input={os.path.abspath(args.input)}",
        "--access_type=1",
        f"--gpu={args.gpu}",
        f"--n_ctrls={args.n_ctrls}",
        f"--threads={args.threads}",
        f"--blk_size={args.blk_size}",
        f"--pages={args.pages}",
        f"--page_size={args.page_size}",
        f"--queue_depth={args.queue_depth}",
        f"--num_queues={args.num_queues}",
        f"--num_blks={args.num_blks}",
        f"--random={args.random}",
        f"--reqs={args.reqs}",
        f"--loffset={args.loffset}",
        f"--ioffset={args.ioffset}",
    ]

    print("LD_LIBRARY_PATH[0:80]:", env["LD_LIBRARY_PATH"][:80], "...", flush=True)
    print("Running:", " ".join(cmd), flush=True)
    if args.dry_run:
        return

    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"error: nvm-readwrite-bench exited {r.returncode}", file=sys.stderr)
        sys.exit(r.returncode)

    if not args.skip_host_sync:
        _sync_host()
        print("Host sync done (os.sync or sync).", flush=True)

    if not args.skip_torch_sync:
        try:
            import torch
        except ImportError:
            print("torch not installed; skipping torch.cuda.synchronize().", flush=True)
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print("torch.cuda.synchronize() done.", flush=True)
            else:
                print("CUDA not available; skipping torch.cuda.synchronize().", flush=True)

    print("OK: write finished (bench per-chunk sync + optional host/GPU sync).", flush=True)


if __name__ == "__main__":
    main()
