#!/usr/bin/env bash
#
# MFU tuning for Qwen3.5-VL 35B-A3B (MoE) — MegatronMIMO with vision CP
#
# Usage:
#   bash tune_35b_a3b_mimo.sh
#
# MIMO uses disjoint rank groups for language and image encoder:
#   --component language=tp=N,etp=M,dp=K,rank_offset=0
#   --component images=tp=1,dp=<world>,rank_offset=0     (colocated)
#
# For MoE models, use etp (or ep) in the component spec for expert parallelism.
# In MIMO, etp maps to the EP dimension in HyperCommGrid.
#
# Results are saved under $OUTPUT_DIR/<tag>/train_node${RANK}.log
# Summary is printed at the end.

set -uo pipefail

# ─── User config ────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-./mfu_tuning_35b_a3b_mimo_cp}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
DATASET_ROOT="${DATASET_ROOT:-/ppusw/datasets_sh}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-35B-A3B}"

SCRIPT="train_qwen35_vl_mimo.py"
TORCHRUN="python -m torch.distributed.run"
mkdir -p "$OUTPUT_DIR"
# ─── Helper ─────────────────────────────────────────────────────────────────
# run_exp <tag> <lang_component_spec> <img_component_spec> [extra_args...]
run_exp() {
    local tag="$1"; shift
    local lang_comp="$1"; shift
    local img_comp="$1"; shift
    local extra_args=("$@")

    local log_dir="${OUTPUT_DIR}/${tag}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/train_node${RANK}.log"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXP : ${tag}"
    echo "  LANG: ${lang_comp}"
    echo "  IMG : ${img_comp}"
    echo "  LOG : ${log_file}"
    echo "════════════════════════════════════════════════════════"

    set +e
    # MODEL_PROFILE_RANGE=10,11 prof.sh "$log_dir/${tag}_$(hostname)"  
    $TORCHRUN \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --rdzv-backend=static \
        --node-rank="$RANK" \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --max-restarts=0 \
        "$SCRIPT" \
        --hf-model "$HF_MODEL" \
        --component "$lang_comp" \
        --component "$img_comp" \
        --output-dir "${log_dir}/tensorboard_node${RANK}" \
        --train-iters "$TRAIN_ITERS" \
        --log-interval "$LOG_INTERVAL" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ $exit_code -ne 0 ]]; then
        echo "[WARN] ${tag} FAILED (exit ${exit_code}), continuing sweep..."
        echo "TUNE_STATUS: FAILED exit=${exit_code}" >> "$log_file"
    else
        echo "TUNE_STATUS: OK" >> "$log_file"
    fi

    sleep 5
}

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_CTA_POLICY=0x1
export NCCL_GRAPH_REGISTER=0
export MEGATRON_LOGGING_LEVEL=20
export NVTE_CPU_OFFLOAD_V1=1 ; export UB_SKIPMC=1 ;
export NVTE_USE_CUTLASS_GROUPED_GEMM=1;
# DeepEP-specific settings, only relevant for 35B-A3B MoE.
export DEEPEP_NUM_MAX_ICN_PEERS=16;
export TRITON_CACHE_DIR=/tmp/triton_cache_$(hostname)
export DEEP_EP_WARMUP_ENABLED=1;
export MEGATRON_VISION_FIXED_LENGTH_ATTENTION=1;
export MIMO_DEBUG=1;

BEST_LANG="${BEST_LANG:-language=tp=2,cp=1,ep=16,dp=8,etp=1,pp=1,rank_offset=0}"
BEST_IMG="${BEST_IMG:-images=tp=1,cp=1,dp=16,rank_offset=0}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1920}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-1080}"
NUM_IMAGES="${NUM_IMAGES:-2}"
SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-1}"
GA="${GA:-8}"

run_exp "35ba3b_${SEQ_LEN}seq_${NUM_IMAGES}img_${IMAGE_SIZE_W}x${IMAGE_SIZE_H}_${MBS}mbs_${GA}ga" \
    "$BEST_LANG" "$BEST_IMG" \
    --moe-dispatcher-type flex --moe-dispatcher-backend deepep \
    --enable-grad-overlap --mbs $MBS --ga $GA \
    --seq-length $SEQ_LEN --num-images $NUM_IMAGES --image-size-w $IMAGE_SIZE_W --image-size-h $IMAGE_SIZE_H \
    --enable-sp --enable-vision-cp --moe-force-load-balancing --enable-cuda-graph-te --enable-deepep-autotune --enable-vision-activation-offload --vision-offload-num-layers 20