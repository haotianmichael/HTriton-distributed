#!/usr/bin/env python3
"""
tp_ep_demo.py —— 一个脚本直接跑 TP / EP 的最小例子（纯形状、随机权重，不下载模型）。

两种实现，同一份数据、同一份权重、同一个参考答案：

  --impl torch   纯 torch.distributed（NCCL）。不依赖 triton_dist，
                 可以直接在 megakernel 环境（Py3.14 / torch 2.15 / cu13.2）里跑。
                 这就是各篇论文里的 "PyTorch + NCCL" 不重叠基线。
  --impl td      在同一位置换成 Triton-distributed 的算子，需要 tdist 环境 + scripts/launch.sh。

两种 case：

  --case tp      一层 TP MLP（gate_up 列切分、down 行切分）
     --tp_mode ar      标准 TP：x 每卡一份 -> GEMM -> GEMM -> AllReduce
     --tp_mode ag_rs   序列并行形式（FLUX 的两个算子）：AllGather+GEMM -> GEMM+ReduceScatter
     --tp_mode gemm_ar 只对 --impl td 有区别：融合的 GEMM+AllReduce kernel
  --case ep      一层 EP MoE：路由 -> dispatch(all-to-all) -> expert FFN -> combine(all-to-all)

启动方式：

  # 1) 纯 torch 基线（megakernel 环境）
  torchrun --nproc_per_node 2 tp_ep_demo.py --case tp --tp_mode ar
  torchrun --nproc_per_node 2 tp_ep_demo.py --case tp --tp_mode ag_rs
  torchrun --nproc_per_node 2 tp_ep_demo.py --case ep

  # 2) Triton-distributed（tdist 环境，在 Triton-distributed 仓库根目录下）
  export NCCL_SOCKET_IFNAME=lo
  bash scripts/launch.sh /path/to/tp_ep_demo.py --impl td --case tp --tp_mode ag_rs
  bash scripts/launch.sh /path/to/tp_ep_demo.py --impl td --case tp --tp_mode gemm_ar
  NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh /path/to/tp_ep_demo.py --impl td --case ep

  # 3) 没有 GPU 时只验证逻辑（自动退到 gloo + float32）
  torchrun --nproc_per_node 2 tp_ep_demo.py --case ep --preset small

单步调试某个 rank（需要 pip install debugpy，然后用 VS Code attach 到 5678 端口）：
  ... tp_ep_demo.py --impl td --case tp --debug_rank 0
"""
import argparse
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))
CUDA = torch.cuda.is_available()

# 形状来自 claude/overlap-benchmark.md；small 只用于在 CPU 上验证逻辑
TP_PRESETS = {
    "llama70b": dict(hidden=8192, inter=28672),
    "qwen3-32b": dict(hidden=5120, inter=25600),
    "small": dict(hidden=256, inter=1024),
}
EP_PRESETS = {
    "qwen3-30b-a3b": dict(hidden=2048, inter=768, experts=128, topk=8),
    "deepseek-v3": dict(hidden=7168, inter=2048, experts=256, topk=8),
    "small": dict(hidden=128, inter=64, experts=16, topk=4),
}


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def sync():
    if CUDA:
        torch.cuda.synchronize()


def log(msg, all_ranks=True):
    if all_ranks or RANK == 0:
        print(f"[rank {RANK}] {msg}", flush=True)


def randw(shape, seed, scale, dtype, device):
    """用独立的 Generator 生成权重/数据：同一个 seed 在每个 rank 上得到同一个张量。"""
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(dtype)


def bench(fn, iters, warmup):
    """返回 (最后一次输出, 平均每次毫秒)。各 rank 靠 collective 本身保持同步。"""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1e3 / iters


class Phases:
    """分阶段计时。开启后每个阶段前后都会 synchronize，所以只用来看占比，不用来报总时间。"""

    def __init__(self):
        self.on = False
        self.ms = {}

    @contextmanager
    def __call__(self, name):
        if not self.on:
            yield
            return
        sync()
        t0 = time.perf_counter()
        yield
        sync()
        self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def measure(self, fn, iters, warmup):
        for _ in range(warmup):
            fn()
        self.on, self.ms = True, {}
        for _ in range(iters):
            fn()
        self.on = False
        return {k: v / iters for k, v in self.ms.items()}


def check(name, out, ref, tol):
    out, ref = out.float(), ref.float()
    err = ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    ok = err < tol
    log(f"check {name}: max|diff|/max|ref| = {err:.2e}  {'PASS' if ok else 'FAIL'} (tol {tol:.0e})")
    assert ok, f"{name} mismatch on rank {RANK}"


def overlap_report(name, comp, comm, seq, method=None):
    """claude/overlap-benchmark.md 里的三个时间、三个数。"""
    line = (f"{name}: 计算单独 {comp:.3f} ms | 通信单独 {comm:.3f} ms | 先算后发(实测) {seq:.3f} ms"
            f" | 加速上限 {(comp + comm) / max(comp, comm):.2f}x")
    if method is not None:
        hidden = (seq - method) / min(comp, comm)
        line += f" | 被测方法 {method:.3f} ms | 实际加速 {seq / method:.2f}x | 藏住的比例 {hidden * 100:.0f}%"
    log(line)


# gloo（CPU）不支持 all_gather_into_tensor / reduce_scatter_tensor，这里给一个等价的退路，
# 只为了没有 GPU 时也能把逻辑跑通；有 GPU 时走的就是 NCCL 的原生 collective。
def all_gather(out, x, group):
    if CUDA:
        dist.all_gather_into_tensor(out, x, group=group)
    else:
        parts = [torch.empty_like(x) for _ in range(group.size())]
        dist.all_gather(parts, x, group=group)
        out.copy_(torch.cat(parts, dim=0))


def reduce_scatter(out, x, group):
    if CUDA:
        dist.reduce_scatter_tensor(out, x, group=group)
    else:
        y = x.clone()
        dist.all_reduce(y, group=group)
        out.copy_(y.chunk(group.size(), dim=0)[group.rank()])


def init_dist(args):
    if args.impl == "td":
        # 必须先 import triton_dist 再 import triton：插件是在 import triton 的那一刻注册的
        import triton_dist  # noqa: F401
        from triton_dist.utils import initialize_distributed
        return initialize_distributed(args.seed)  # 建 NCCL group + 初始化 NVSHMEM（一卡一个 PE）
    if CUDA:
        torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl" if CUDA else "gloo")
    return dist.group.WORLD


def finalize_dist(args):
    if args.impl == "td":
        from triton_dist.utils import finalize_distributed
        finalize_distributed()
    else:
        dist.destroy_process_group()


def td_barrier():
    from triton_dist.utils import nvshmem_barrier_all_on_stream
    nvshmem_barrier_all_on_stream()
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# TP：一层 MLP
# --------------------------------------------------------------------------- #
class _Linear:
    """只带 weight 的假 Linear，长得像 HF 的 nn.Linear（triton_dist 的 TP_MLP 只读 .weight / .bias）。"""

    def __init__(self, weight):
        self.weight = weight
        self.bias = None


class DenseMLP:
    """完整（未切分）的 SwiGLU MLP：down(silu(gate(x)) * up(x))。每个 rank 都从同一个 seed 生成，
    一份用来切分成 TP 权重，一份用来算参考答案。"""

    def __init__(self, hidden, inter, dtype, device, seed):
        self.gate_proj = _Linear(randw((inter, hidden), seed + 1, hidden**-0.5, dtype, device))
        self.up_proj = _Linear(randw((inter, hidden), seed + 2, hidden**-0.5, dtype, device))
        self.down_proj = _Linear(randw((hidden, inter), seed + 3, inter**-0.5, dtype, device))
        self.act_fn = torch.nn.SiLU()

    @torch.inference_mode()
    def __call__(self, x):
        h = self.act_fn(F.linear(x, self.gate_proj.weight)) * F.linear(x, self.up_proj.weight)
        return F.linear(h, self.down_proj.weight)


class TorchTP:
    """PyTorch + NCCL 的 TP MLP。gate/up 按输出维（行）切，down 按输入维（列）切。"""

    def __init__(self, mlp: DenseMLP, group):
        P, r = group.size(), group.rank()
        self.group = group
        gate = mlp.gate_proj.weight.chunk(P, dim=0)[r]
        up = mlp.up_proj.weight.chunk(P, dim=0)[r]
        self.gate_up = torch.cat((gate, up), dim=0).contiguous()  # [2*I/P, H]
        self.down = mlp.down_proj.weight.chunk(P, dim=1)[r].contiguous()  # [H, I/P]

    def local(self, x):  # 两个 GEMM，没有任何通信
        g, u = F.linear(x, self.gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down)

    @torch.inference_mode()
    def fwd_ar(self, x):  # x: [M, H]，每卡一份完整输入
        z = self.local(x)
        dist.all_reduce(z, group=self.group)  # <- 依赖型通信：等 GEMM 全算完才开始
        return z

    @torch.inference_mode()
    def fwd_ag_rs(self, x_local):  # x_local: [M/P, H]
        P = self.group.size()
        x = torch.empty((x_local.shape[0] * P, x_local.shape[1]), dtype=x_local.dtype, device=x_local.device)
        all_gather(x, x_local, self.group)  # <- GEMM 要等 AllGather 全到
        z = self.local(x)
        out = torch.empty_like(x_local)
        reduce_scatter(out, z, self.group)  # <- ReduceScatter 要等 GEMM 全算完
        return out


def run_tp(args, group, dtype, device, tol):
    cfg = TP_PRESETS[args.preset or "llama70b"]
    H, I, M = cfg["hidden"], cfg["inter"], args.tokens
    P = group.size()
    assert M % P == 0 and I % P == 0
    mode = args.tp_mode
    log(f"TP MLP: M={M} H={H} I={I} P={P} mode={mode} impl={args.impl} dtype={dtype}", all_ranks=False)

    mlp = DenseMLP(H, I, dtype, device, args.seed)
    x = randw((M, H), args.seed + 100, 0.25, dtype, device)  # 所有 rank 相同
    golden = mlp(x)
    x_local = x.chunk(P, dim=0)[RANK].contiguous()
    golden_local = golden.chunk(P, dim=0)[RANK]

    tp = TorchTP(mlp, group)
    sp = mode == "ag_rs"  # 序列并行：输入输出都只有 M/P 行
    torch_fn = (lambda: tp.fwd_ag_rs(x_local)) if sp else (lambda: tp.fwd_ar(x))
    ref = golden_local if sp else golden

    check("torch", torch_fn(), ref, tol)
    _, seq_ms = bench(torch_fn, args.iters, args.warmup)

    # 计算单独 / 通信单独
    _, comp_ms = bench(lambda: tp.local(x), args.iters, args.warmup)
    buf = torch.zeros_like(golden)  # 全 0：反复原地 all_reduce 也不会溢出
    if sp:
        ag_out, rs_out = torch.empty_like(x), torch.empty_like(x_local)

        def comm_fn():
            all_gather(ag_out, x_local, group)
            reduce_scatter(rs_out, buf, group)
    else:

        def comm_fn():
            dist.all_reduce(buf, group=group)

    _, comm_ms = bench(comm_fn, args.iters, args.warmup)

    td_ms = None
    if args.impl == "td":
        import triton
        from triton_dist.layers.nvidia.tp_mlp import TP_MLP

        td = TP_MLP(rank=RANK, world_size=P, group=group)
        td._init_parameters(mlp, verbose=RANK == 0)  # 和 TorchTP 完全相同的切分
        if mode == "ag_rs":
            # ag_gemm（07 教程）+ gemm_rs（08 教程）
            triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
            td._init_ctx(max_M=M, ag_intranode_stream=torch.cuda.Stream(priority=-1),
                         ag_internode_stream=torch.cuda.Stream())
            td_fn = lambda: td.dist_triton_fwd(x_local, autotune=args.autotune)  # noqa: E731
        elif mode == "gemm_ar":
            # 融合的 GEMM+AllReduce（kernels/nvidia/gemm_allreduce.py）
            td._init_gemm_ar_ctx(max_M=M, dtype=dtype)
            td_fn = lambda: td.dist_triton_gemm_ar_fwd(x)  # noqa: E731
        else:
            # GEMM 用 torch，只把 NCCL AllReduce 换成 triton_dist 的 AllReduce kernel（不重叠）
            from triton_dist.kernels.nvidia.allreduce import get_auto_allreduce_method
            method = get_auto_allreduce_method(M * H * dtype.itemsize)
            log(f"triton_dist allreduce method = {method.name}", all_ranks=False)
            td._init_AR_ctx(max_M=M, method=method, dtype=dtype)
            td_fn = lambda: td.dist_triton_AR_fwd(x)  # noqa: E731

        td_barrier()
        check("triton_dist", td_fn(), ref, tol)
        td_barrier()
        _, td_ms = bench(td_fn, args.iters, args.warmup)
        td_barrier()
        td.finalize()

    overlap_report(f"TP[{mode}]", comp_ms, comm_ms, seq_ms, td_ms)


# --------------------------------------------------------------------------- #
# EP：一层 MoE
# --------------------------------------------------------------------------- #
class ExpertBank:
    """每个 expert 的权重由 (seed, expert_id) 唯一决定：本 rank 只常驻自己那一段 expert，
    做 --check 时任何 rank 都能临时重建任意一个 expert 来算参考答案。"""

    def __init__(self, hidden, inter, experts, dtype, device, seed, group):
        self.H, self.I, self.G = hidden, inter, experts
        self.dtype, self.device, self.seed = dtype, device, seed
        P, r = group.size(), group.rank()
        assert experts % P == 0
        self.per_rank = experts // P
        self.lo, self.hi = r * self.per_rank, (r + 1) * self.per_rank
        self.router = randw((experts, hidden), seed + 7, hidden**-0.5, dtype, device)  # [G, H]
        ws = [self.expert(e) for e in range(self.lo, self.hi)]
        self.gate_up = torch.stack([w[0] for w in ws])  # [G/P, 2I, H]
        self.down = torch.stack([w[1] for w in ws])  # [G/P, H, I]

    def expert(self, e):
        gate_up = randw((2 * self.I, self.H), self.seed * 100003 + 2 * e, self.H**-0.5, self.dtype, self.device)
        down = randw((self.H, self.I), self.seed * 100003 + 2 * e + 1, self.I**-0.5, self.dtype, self.device)
        return gate_up, down

    @staticmethod
    def ffn(x, gate_up, down):
        g, u = F.linear(x, gate_up).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, down)

    def route(self, x, topk):
        probs = F.softmax(F.linear(x, self.router).float(), dim=-1)
        w, ids = torch.topk(probs, topk, dim=-1)
        w = (w / w.sum(dim=-1, keepdim=True)).contiguous()
        return w, ids.to(torch.int32).contiguous()  # [T, topk] float32 / int32

    def local_experts(self, tokens, counts):
        """tokens 已经按本 rank 的 expert 排好序，counts[e] 是每个 expert 的行数。
        两种实现共用这一个函数，所以两边的差别只在通信。"""
        out = torch.empty_like(tokens)
        s = 0
        for e, c in enumerate(counts):
            if c:
                out[s:s + c] = self.ffn(tokens[s:s + c], self.gate_up[e], self.down[e])
                s += c
        return out

    @torch.inference_mode()
    def reference(self, x, w, ids):
        """不做任何通信的参考答案：out[t] = sum_k w[t,k] * FFN_{ids[t,k]}(x[t])。"""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in range(self.G):
            rows, cols = (ids == e).nonzero(as_tuple=True)
            if rows.numel():
                y = self.ffn(x[rows], *self.expert(e)).float()
                out.index_add_(0, rows, y * w[rows, cols].unsqueeze(-1))
        return out


class TorchEP:
    """PyTorch + NCCL 的 EP MoE：两次 all_to_all 之间夹着 expert 计算。"""

    def __init__(self, bank: ExpertBank, group, topk):
        self.bank, self.group, self.topk = bank, group, topk
        self.ph = Phases()

    @torch.inference_mode()
    def fwd(self, x, w, ids):
        bank, group, ph = self.bank, self.group, self.ph
        P, G, per_rank, topk = group.size(), bank.G, bank.per_rank, self.topk
        T, H = x.shape

        with ph("dispatch"):
            flat = ids.flatten().long()
            order = torch.argsort(flat, stable=True)  # 按全局 expert 排序 => 自动按目标 rank 分段
            src_token = order // topk
            send_tok = x[src_token]
            send_w = w.flatten()[order]
            my_splits = torch.bincount(flat, minlength=G)  # [G]
            # 每个 rank 都要知道 "谁给我的哪个 expert 发了多少"：先 all_gather 一个 [P, G] 的小表
            full = torch.empty((P, G), dtype=my_splits.dtype, device=x.device)
            all_gather(full.view(-1), my_splits, group)
            mine = full[:, bank.lo:bank.hi]  # [P, G/P]：行 = 来源 rank，列 = 本地 expert
            in_splits = my_splits.view(P, per_rank).sum(dim=1).tolist()
            out_splits = mine.sum(dim=1).tolist()
            n_recv = sum(out_splits)
            recv_tok = torch.empty((n_recv, H), dtype=x.dtype, device=x.device)
            recv_w = torch.empty((n_recv, ), dtype=w.dtype, device=x.device)
            dist.all_to_all_single(recv_tok, send_tok, out_splits, in_splits, group=group)
            dist.all_to_all_single(recv_w, send_w, out_splits, in_splits, group=group)

        with ph("expert"):
            # 收到的行是 "先按来源 rank、再按 expert" 排的，重排成按 expert 连续
            eid = torch.repeat_interleave(torch.arange(per_rank, device=x.device).repeat(P), mine.flatten())
            perm = torch.argsort(eid, stable=True)
            y = bank.local_experts(recv_tok[perm], mine.sum(dim=0).tolist())
            y = (y.float() * recv_w[perm].unsqueeze(-1)).to(x.dtype)
            send_back = torch.empty_like(y)
            send_back[perm] = y

        with ph("combine"):
            back = torch.empty((T * topk, H), dtype=x.dtype, device=x.device)
            dist.all_to_all_single(back, send_back, in_splits, out_splits, group=group)
            out = torch.zeros((T, H), dtype=x.dtype, device=x.device)
            out.index_add_(0, src_token, back)  # 每个 token 把 topk 份结果加起来
        return out


class TritonDistEP:
    """同样的三段，但 dispatch / combine 换成 triton_dist 的 EPAll2AllLayer：
    kernel 里每个 warp 直接把 token put 到目标 rank 的对称内存（kernels/nvidia/ep_a2a_intra_node.py）。
    这条路径只用到本地 atomic + 远端普通写，PCIe（无 P2P atomic）机器也能跑。"""

    def __init__(self, bank: ExpertBank, group, topk, max_tokens, num_sm):
        from triton_dist.layers.nvidia import EPAll2AllLayer
        self.bank, self.group, self.topk = bank, group, topk
        self.ph = Phases()
        self.layer = EPAll2AllLayer(group, max_tokens, bank.H, topk, RANK, bank.G, LOCAL_WORLD_SIZE, group.size(),
                                    dtype=bank.dtype, weight_dtype=torch.float32, num_sm=num_sm)

    @torch.inference_mode()
    def fwd(self, x, w, ids):
        bank, layer, ph = self.bank, self.layer, self.ph
        with ph("dispatch"):
            # 返回的行已经按 (本地 expert, 来源 rank) 连续排好，可以直接喂 grouped GEMM
            tok, tok_w, layout = layer.dispatch(x, ids, weight=w)
        with ph("expert"):
            # 每个本地 expert 收到多少行。EPAll2AllLayer 在 dispatch 结束时把内部的 splits 表清零了，
            # 这里用它自带的 splits_golden（一次很小的 NCCL all_gather）重新拿一份。
            counts = layer.splits_golden(ids, bank.G)[:, bank.lo:bank.hi].sum(dim=0).tolist()
            assert sum(counts) == tok.shape[0], (sum(counts), tok.shape)
            y = bank.local_experts(tok, counts)
            y = (y.float() * tok_w.unsqueeze(-1)).to(x.dtype)
        with ph("combine"):
            out = layer.combine(y, layout)  # kernel 内按 token 把 topk 份结果加起来
        return out

    def finalize(self):
        self.layer.finalize()


def run_ep(args, group, dtype, device, tol):
    cfg = EP_PRESETS[args.preset or "qwen3-30b-a3b"]
    H, I, G, topk, T = cfg["hidden"], cfg["inter"], cfg["experts"], cfg["topk"], args.tokens
    log(f"EP MoE: tokens/rank={T} H={H} I={I} experts={G} topk={topk} P={group.size()} impl={args.impl} dtype={dtype}",
        all_ranks=False)

    bank = ExpertBank(H, I, G, dtype, device, args.seed, group)
    x = randw((T, H), args.seed + 1000 + RANK, 1.0, dtype, device)  # 每个 rank 自己的 token
    w, ids = bank.route(x, topk)

    ep = TorchEP(bank, group, topk)
    ref = bank.reference(x, w, ids) if args.check else None
    out = ep.fwd(x, w, ids)
    if args.check:
        check("torch", out, ref, tol)
    _, seq_ms = bench(lambda: ep.fwd(x, w, ids), args.iters, args.warmup)
    ph = ep.ph.measure(lambda: ep.fwd(x, w, ids), args.iters, args.warmup)
    comp_ms, comm_ms = ph["expert"], ph["dispatch"] + ph["combine"]
    log(f"torch  分阶段: dispatch {ph['dispatch']:.3f} | expert {ph['expert']:.3f} | combine {ph['combine']:.3f} ms")

    td_ms = None
    if args.impl == "td":
        td = TritonDistEP(bank, group, topk, max_tokens=T, num_sm=args.ep_num_sm)
        td_barrier()
        out = td.fwd(x, w, ids)
        if args.check:
            check("triton_dist", out, ref, tol)
        td_barrier()
        _, td_ms = bench(lambda: td.fwd(x, w, ids), args.iters, args.warmup)
        td_barrier()
        ph = td.ph.measure(lambda: td.fwd(x, w, ids), args.iters, args.warmup)
        log(f"td     分阶段: dispatch {ph['dispatch']:.3f} | expert {ph['expert']:.3f} | combine {ph['combine']:.3f} ms")
        td_barrier()
        td.finalize()

    overlap_report("EP", comp_ms, comm_ms, seq_ms, td_ms)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["tp", "ep"], default="tp")
    p.add_argument("--impl", choices=["torch", "td"], default="torch")
    p.add_argument("--tp_mode", choices=["ar", "ag_rs", "gemm_ar"], default="ar")
    p.add_argument("--preset", default=None,
                   help=f"tp: {list(TP_PRESETS)}（默认 llama70b）; ep: {list(EP_PRESETS)}（默认 qwen3-30b-a3b）")
    p.add_argument("--tokens", type=int, default=None, help="tp: 总 token 数 M（默认 4096）; ep: 每个 rank 的 token 数（默认 2048）")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", default=True, action=argparse.BooleanOptionalAction, help="和无通信的参考答案比对")
    p.add_argument("--autotune", default=False, action=argparse.BooleanOptionalAction,
                   help="td ag_rs：是否跑 triton_dist 的 autotune（调试时关掉，报性能时打开）")
    p.add_argument("--ep_num_sm", type=int, default=20, help="td ep：dispatch/combine kernel 用多少个 SM")
    p.add_argument("--debug_rank", type=int, default=-1, help="这个 rank 启动后等 debugpy attach")
    p.add_argument("--debug_port", type=int, default=5678)
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug_rank == RANK:
        import debugpy
        debugpy.listen(("0.0.0.0", args.debug_port))
        log(f"waiting for debugger on port {args.debug_port} ...")
        debugpy.wait_for_client()

    if args.impl == "td":
        assert CUDA, "--impl td 需要 GPU 和 triton_dist"
    if args.impl == "torch" and args.tp_mode == "gemm_ar":
        args.tp_mode = "ar"  # 对 torch 来说 gemm_ar 就是 ar
    if not CUDA and args.preset is None:
        args.preset = "small"
    if args.tokens is None:
        args.tokens = (4096 if args.case == "tp" else 2048) if CUDA else 64

    group = init_dist(args)
    device = torch.device("cuda", torch.cuda.current_device()) if CUDA else torch.device("cpu")
    dtype = torch.bfloat16 if CUDA else torch.float32
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    try:
        (run_tp if args.case == "tp" else run_ep)(args, group, dtype, device, tol)
    finally:
        sync()
        finalize_dist(args)


if __name__ == "__main__":
    main()