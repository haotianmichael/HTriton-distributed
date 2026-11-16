#!/usr/bin/env bash
# =============================================================================
# Triton-distributed（main = Triton 3.7.1 的 out-of-tree plugin）免 Docker、免 sudo 安装。
#
# 为什么不能装进 megakernel（Py3.14 / torch 2.15 / cu13.2）：
#   - python/setup.py 写死 python_requires ">=3.9,<3.14"；
#   - 它依赖的 nvshmem4py-cu12==0.3.0 没有 cp314 的 wheel；
#   - 编译时会卸掉 torch 自带的 triton，换成自己打补丁的 3.7.1（torch 2.15 要 triton~=3.9），
#     并装 nvidia-nvshmem-cu12==3.6.5（torch 2.15 带的是 nvshmem-cu13 3.7.2）。
# 所以单独建一个 conda 环境 tdist（Python 3.12 + torch 2.8.0/cu128），megakernel 完全不动。
# torch 的 CUDA 12.8 运行库在 wheel 里，系统驱动（支持到 CUDA 13.0）向下兼容，不需要系统 CUDA。
#
# zlib 也装进 conda 环境（2026-11-16 修）：Triton 预编译 LLVM 的 LLVMSupport / lldELF 链接 ZLIB::ZLIB。
# 用 conda 的 x86_64-conda-linux-gnu-g++ 时，CMake 不搜 Ubuntu 的 /usr/lib/x86_64-linux-gnu，
# 只找到 /usr/include/zlib.h、找不到 libz.so，configure 阶段就报 "ZLIB::ZLIB but the target was not found"。
#
# 用法：  bash setup_tdist_env.sh [Triton-distributed 仓库目录，默认 ~/MegaKernel/Triton-distributed]
#         （传它的上一级目录也行）
# 环境变量：ENV_NAME（默认 tdist） MAX_JOBS（默认 内存GB/4） SKIP_TESTS=1（只装不测）
# =============================================================================
set -eo pipefail
ENV_NAME=${ENV_NAME:-tdist}
WORKDIR=${1:-$HOME/MegaKernel/Triton-distributed}
log() { echo -e "\n\033[1;34m==> $*\033[0m"; }
die() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }
# 兼容旧用法（传的是 ~/MegaKernel）
[ -f "${WORKDIR}/Triton-distributed/python/setup.py" ] && WORKDIR="${WORKDIR}/Triton-distributed"
cd "${WORKDIR}" 2>/dev/null || die "目录不存在：${WORKDIR}"
[ -f python/setup.py ] && [ -f scripts/build_triton.sh ] || die "${WORKDIR} 不像 Triton-distributed 仓库根目录"
WORKDIR=$(pwd)

log "0. 宿主机检查"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi"
command -v git >/dev/null || die "需要 git"
command -v conda >/dev/null || die "需要 conda 在 PATH 里"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi topo -p2p a || true   # 你的机器这里是 NS：代码会自动走非 atomic 的 barrier
glibc_ver=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
[ "$(printf '%s\n%s\n' 2.28 "${glibc_ver}" | sort -V | head -1)" = "2.28" ] \
  || die "glibc ${glibc_ver} < 2.28：拿不到预编译 LLVM / torch wheel"
if [ -z "${MAX_JOBS}" ]; then
  MAX_JOBS=$(( $(free -g | awk '/^Mem:/{print $2}') / 4 ))
  [ "${MAX_JOBS}" -gt "$(nproc)" ] && MAX_JOBS=$(nproc)
  [ "${MAX_JOBS}" -lt 1 ] && MAX_JOBS=1
fi
export MAX_JOBS   # 文档里的 126 很容易把内存打爆
echo "glibc ${glibc_ver} | MAX_JOBS=${MAX_JOBS} | 仓库 ${WORKDIR}"

log "1. conda 环境 ${ENV_NAME}（Python 3.12 + conda 自带的 g++ 13 + zlib，不依赖系统编译器和系统库）"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" -c conda-forge python=3.12 gxx_linux-64=13 make zlib
fi
conda activate "${ENV_NAME}"
# 旧版脚本建的环境里没有 zlib 头文件和 libz.so，补上
if [ ! -f "${CONDA_PREFIX}/include/zlib.h" ] || [ ! -e "${CONDA_PREFIX}/lib/libz.so" ]; then
  conda install -y -c conda-forge zlib
fi
# 让 Triton 和 triton_dist 插件两次 CMake configure（都会 find_package(MLIR) -> find_package(ZLIB)）
# 都先在 conda 环境里找
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
echo "CXX=${CXX:-<unset>}"; "${CXX:-g++}" --version | head -1
echo "zlib: $(ls "${CONDA_PREFIX}"/lib/libz.so.*.* 2>/dev/null | head -1)"

log "2. torch 2.8.0（cu128；不带 nvshmem 依赖，和 triton_dist 要的 nvshmem-cu12 3.6.5 不冲突）"
pip install "torch==2.8.0"

log "3. 构建工具"
pip install "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" lit wheel "setuptools>=69"

log "4. 编译（先编打补丁的 Triton 3.7.1，再编插件 libtriton_dist.so；会联网下载预编译 LLVM）"
# 上一次 configure 失败会在 CMakeCache 里留下 ZLIB_INCLUDE_DIR=/usr/include 和 libz NOTFOUND，
# 不清掉的话 CMake 会继续用缓存值。只删 ZLIB 不在 conda 环境里的那份缓存，正常重跑不受影响。
while IFS= read -r cache; do
  if ! grep -qF "ZLIB_INCLUDE_DIR:PATH=${CONDA_PREFIX}/" "${cache}"; then
    echo "删除上次失败留下的 ${cache}"; rm -f "${cache}"
  fi
done < <(find 3rdparty/triton/build -maxdepth 2 -name CMakeCache.txt 2>/dev/null)
constraint=$(mktemp); echo 'numpy<2' > "${constraint}"
export USE_TRITON_DISTRIBUTED_AOT=0
pip install -c "${constraint}" -e "python[build,tests,tutorials]" --no-build-isolation --use-pep517 -v
rm -f "${constraint}"
pip install debugpy   # tp_ep_demo.py --debug_rank 用

python -c "import triton_dist, triton, torch; print('triton', triton.__version__, '| torch', torch.__version__)"
python -c "from triton_dist.utils import supports_p2p_native_atomic, has_fullmesh_nvlink; \
print('p2p_native_atomic =', supports_p2p_native_atomic(), '| fullmesh_nvlink =', has_fullmesh_nvlink())"

[ "${SKIP_TESTS:-0}" = "1" ] && { log "SKIP_TESTS=1，结束"; exit 0; }

log "5. 冒烟测试（只挑单机 PCIe、无 P2P atomic 能跑的）"
# setenv.sh 把 NVSHMEM 的 bootstrap 网卡写死成 eth0；launch.sh 在 NCCL_SOCKET_IFNAME 有值时改用它。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
T=python/triton_dist/test/nvidia
declare -a results
run() {
  local name=$1; shift
  echo -e "\n--- ${name}: $*"
  if "$@"; then results+=("PASS  ${name}"); else results+=("FAIL  ${name}"); fi
}
run nvshmem_api    bash scripts/launch.sh $T/test_nvshmem_api.py
run notify_wait    bash scripts/launch.sh $T/test_distributed_wait.py --case correctness
run tutorial_01    bash scripts/launch.sh tutorials/01-distributed-notify-wait.py
run tutorial_02    bash scripts/launch.sh tutorials/02-intra-node-allgather.py
run ag_gemm        bash scripts/launch.sh $T/test_ag_gemm.py --case check --no-autotune
run gemm_rs        bash scripts/launch.sh $T/test_gemm_rs.py -M 8192 -N 8192 -K 14336 --check
run tutorial_08    bash scripts/launch.sh tutorials/08-overlapping-gemm-reduce-scatter.py
run ep_a2a         env NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh $T/test_ep_a2a.py -M 4096 -N 2048 --topk 8 -G 128 --check
echo; printf '%s\n' "${results[@]}"