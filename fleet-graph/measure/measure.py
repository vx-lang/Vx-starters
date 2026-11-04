"""Measure the primitives that cost.vx prices, on real GPUs.

One process per GPU, on one node or several:

  torchrun --nproc-per-node 8 measure.py > out-1node.txt
  torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint $HEAD:29500 measure.py > out-2node.txt

Rank 0 prints one line per measurement: `measured <name> key=value ...`. calibrate.py reads those
lines and says what to change in cost.vx and hw.vx. Run it on a quiet machine: another job on the
GPUs, or a power cap, moves every number. The header line records what the machine says it is.

NOT YET RUN ON A GPU. It was written on a machine without one; the first run is the test.

What is measured, and the factor of cost.vx it feeds:

  hbm_copy, hbm_read   achieved HBM bytes/s of a big copy and of a big reduction   -> mem_eff()
  gemm_bf16, gemm_fp8  achieved FLOP/s of a large matmul                           -> mfu()
  launch               microseconds of a short kernel launched back to back        -> layer_overhead()
  a2a_local            all-to-all inside a node: time against bytes per GPU        -> comm_eff(), nvlink
  a2a_world            the same over every rank (across nodes when there are two)  -> comm_eff(), cross, lat
  p2p                  send of a microbatch's hidden states from rank 0 to rank `gpn`
                       (the first GPU of the next node)                            -> pipeline.vx send_time()
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def sync():
    torch.cuda.synchronize()


def timed(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def say(rank, name, **kv):
    if rank == 0:
        print("measured " + name + " " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def fit_line(points):
    """Least-squares seconds = lat + bytes / bw over (bytes, seconds) points -> (lat, bw)."""
    n = len(points)
    sx = sum(b for b, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum(b * b for b, _ in points)
    sxy = sum(b * t for b, t in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, 1.0 / slope


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    say(rank, "machine", gpu=props.name.replace(" ", "_"), sm=props.multi_processor_count,
        hbm_bytes=props.total_memory, world=world, gpus_per_node=gpn, torch=torch.__version__,
        cuda=torch.version.cuda)

    # Memory bandwidth: a copy reads and writes, a reduction only reads.
    n = 1 << 30  # 4 GiB of f32
    a = torch.empty(n, device=dev, dtype=torch.float32).normal_()
    b = torch.empty_like(a)
    t = timed(lambda: b.copy_(a))
    say(rank, "hbm_copy", bytes_per_s=2 * 4 * n / t)
    t = timed(lambda: a.sum())
    say(rank, "hbm_read", bytes_per_s=4 * n / t)
    del a, b

    # Arithmetic: large square matmuls.
    m = 8192
    x = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    y = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    t = timed(lambda: x @ y)
    say(rank, "gemm_bf16", flops_per_s=2 * m**3 / t)
    if hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
        try:
            xf = x.to(torch.float8_e4m3fn)
            yf = y.t().contiguous().to(torch.float8_e4m3fn).t()
            one = torch.tensor(1.0, device=dev)
            t = timed(lambda: torch._scaled_mm(xf, yf, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            say(rank, "gemm_fp8", flops_per_s=2 * m**3 / t)
        except Exception as e:  # the signature of _scaled_mm has changed between versions
            say(rank, "gemm_fp8", error=type(e).__name__)
    del x, y

    # Launch overhead: a tiny kernel, back to back.
    z = torch.zeros(1, device=dev)
    t = timed(lambda: [z.add_(1.0) for _ in range(1000)], warm=2, iters=5)
    say(rank, "launch", us=t / 1000 * 1e6)

    # All-to-all. `per_gpu` is the bytes each GPU sends in all (split evenly over the ranks of the
    # group), as cost.vx's all_to_all_bytes counts them.
    def a2a(group, label):
        size = dist.get_world_size(group)
        pts = []
        for kb in (16, 64, 256, 1024, 4096, 16384, 65536):
            per_gpu = kb * 1024
            per_gpu -= per_gpu % (size * 2)
            buf = torch.empty(per_gpu // 2, device=dev, dtype=torch.bfloat16)
            out = torch.empty_like(buf)
            dist.barrier(group)
            t = timed(lambda: dist.all_to_all_single(out, buf, group=group))
            t = torch.tensor([t], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
            pts.append((per_gpu, t.item()))
            say(rank, label + "_point", ranks=size, bytes_per_gpu=per_gpu, seconds=t.item())
        lat, bw = fit_line(pts[2:])  # the small sizes are latency, not a line
        say(rank, label, ranks=size, lat_s=lat, bytes_per_s=bw)

    node = rank // gpn
    local = dist.new_group(list(range(node * gpn, (node + 1) * gpn)))
    a2a(local, "a2a_local")
    if world > gpn:
        a2a(dist.group.WORLD, "a2a_world")

    # Point to point: the handoff of a pipeline. Rank 0 sends to the same GPU of the next node (or,
    # on one node, to the next GPU, which is NVLink).
    peer = gpn if world > gpn else 1
    pts = []
    for kb in (16, 64, 256, 1024, 4096, 16384):
        buf = torch.empty(kb * 512, device=dev, dtype=torch.bfloat16)

        def hop():
            if rank == 0:
                dist.send(buf, peer)
            elif rank == peer:
                dist.recv(buf, 0)

        dist.barrier()
        t = timed(hop)
        t = torch.tensor([t], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        pts.append((kb * 1024, t.item()))
    lat, bw = fit_line(pts[2:])
    say(rank, "p2p", peer=peer, lat_s=lat, bytes_per_s=bw)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
