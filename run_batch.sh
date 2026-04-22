#!/usr/bin/env bash
# 4x RTX 4090 data-parallel batch driver for LTX-2 ti2vid_two_stages (fp8).
# 每张卡起一个长驻 Python 进程，进程内用 StateDictRegistry 缓存所有 state_dict，
# 第二条 prompt 开始零磁盘 IO、零 pin_memory 重复开销。
#
# 用法:
#   cd /home/yanzhang/LTX-2
#   ./run_batch.sh                       # 默认用 ./prompts.json
#   ./run_batch.sh my_prompts.json       # 自定义 prompt 清单

set -euo pipefail

PROMPTS_JSON="${1:-prompts.json}"
NUM_GPUS="${NUM_GPUS:-4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----- runtime env -----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# 避免 4 个 worker 抢同一组 CPU 核 —— 每 worker 拿 1/N 的线程
TOTAL_CPUS="$(nproc)"
export OMP_NUM_THREADS=$(( TOTAL_CPUS / NUM_GPUS ))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# ----- output dirs -----
mkdir -p "${REPO_ROOT}/outputs" "${REPO_ROOT}/logs"

# ----- sanity check -----
if [[ ! -f "${REPO_ROOT}/${PROMPTS_JSON}" ]]; then
  echo "prompts file not found: ${REPO_ROOT}/${PROMPTS_JSON}" >&2
  exit 1
fi

echo "launching ${NUM_GPUS} workers on $(date)"
pids=()
for gpu in $(seq 0 $(( NUM_GPUS - 1 ))); do
  CUDA_VISIBLE_DEVICES="$gpu" \
  python "${REPO_ROOT}/batch_driver.py" \
    --prompts-json "${REPO_ROOT}/${PROMPTS_JSON}" \
    --worker-id "$gpu" \
    --num-workers "$NUM_GPUS" \
    --height 768 --width 1024 \
    --num-frames 97 --frame-rate 24 \
    --num-inference-steps 30 \
    --streaming-prefetch-count 8 \
    --max-batch-size 4 \
    > "${REPO_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  worker gpu=${gpu} pid=${pids[-1]} -> logs/gpu${gpu}.log"
done

# 任意一个 worker 崩溃就整体失败
trap 'echo "interrupted"; kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "worker pid=${pid} FAILED" >&2
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "one or more workers failed — see logs/gpu*.log" >&2
  exit 1
fi

echo "all workers finished on $(date)"
ls -la "${REPO_ROOT}/outputs"
