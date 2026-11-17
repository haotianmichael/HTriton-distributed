#!/usr/bin/env python
"""在 debugpy 下运行 Triton-distributed 的测试脚本，给 VS Code attach 用。

放在仓库根目录。用法（tdist 环境）：

  # 两个进程，只调 rank 0（rank 1 正常跑，到需要 rank 0 配合的地方会自己等）
  DEBUG_RANK=0 bash scripts/launch.sh td_debug.py <脚本.py> [脚本参数...]

  # 两个 rank 都等 VS Code：rank r 监听 5678 + r
  DEBUG_RANK=all bash scripts/launch.sh td_debug.py <脚本.py> [脚本参数...]

  # 单进程
  source scripts/setenv.sh
  DEBUG_RANK=0 python td_debug.py <脚本.py> [脚本参数...]

不设 DEBUG_RANK 时等价于直接运行脚本。端口起点可用 DEBUG_PORT 改（默认 5678）。
只监听 127.0.0.1：VS Code 用 Remote-SSH 连到这台机器后 attach 到 localhost 即可。
"""
import os
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    target = os.path.abspath(sys.argv[1])
    if not os.path.isfile(target):
        sys.exit(f"[td_debug] 找不到脚本：{target}")

    # torchrun 给每个进程设 LOCAL_RANK / RANK；单进程时都没有，当作 rank 0
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    want = os.environ.get("DEBUG_RANK", "").strip().lower()
    if want and (want == "all" or rank in {int(x) for x in want.split(",") if x.strip()}):
        # 只是关掉 Python 3.11+ 下 debugpy 关于 frozen modules 的一段提示，不影响断点
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        import debugpy

        port = int(os.environ.get("DEBUG_PORT", "5678")) + rank
        debugpy.listen(("127.0.0.1", port))
        print(f"[td_debug] rank {rank} (pid {os.getpid()}) 在等 VS Code attach 到端口 {port} ...", flush=True)
        debugpy.wait_for_client()
        print(f"[td_debug] rank {rank} 已连接，开始运行 {os.path.basename(target)}", flush=True)

    # 让被调试的脚本看到的 argv 和 sys.path[0]，与 `python 脚本.py 参数...` 时完全一样
    sys.argv = [target] + sys.argv[2:]
    sys.path[0] = os.path.dirname(target)
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()