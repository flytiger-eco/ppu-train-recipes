# Qwen3.5-VL Best Practices

Training and MFU tuning recipes for **Qwen3.5-VL** on **PPU 890P**.

## Performance Results

### PPU 890P

Experiment Setup:
- 1 node for 4B / 9B dense models (8 PPU 890P per node)
- 2 nodes for 27B dense model (16 PPU 890P in total)
- Image resolution: 1920x1080
- Precision: bf16
- Metrics are averaged over the last 10 iterations of each run

#### Dense Models

| Model | SEQ_LEN | NUM_IMAGES | NNODES | NPROC | MBS | GA | GBS | DP | Step time (sec) | Per GPU TF | Tokens/s | Tokens/s/GPU | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4B  | 4096 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 5.07  | 170.47 | 51784 | 6473 | |
| 4B  | 4096 | 2 | 1 | 8 | 1 | 8 | 64  | 8  | 10.56 | 134.38 | 24823 | 3103 | |
| 4B  | 8192 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 9.74  | 182.71 | 53849 | 6731 | |
| 4B  | 8192 | 4 | 1 | 8 | 1 | 8 | 64  | 8  | 19.16 | 150.90 | 27363 | 3420 | |
| 9B  | 4096 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 7.26  | 219.84 | 36098 | 4512 | |
| 9B  | 4096 | 2 | 1 | 8 | 1 | 8 | 64  | 8  | 14.16 | 164.19 | 18518 | 2315 | |
| 9B  | 8192 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 14.07 | 230.69 | 37263 | 4658 | |
| 27B | 4096 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 13.11 | 196.31 | 20000 | 2500 | |
| 27B | 4096 | 2 | 1 | 8 | 1 | 8 | 64  | 8  | 17.53 | 167.62 | 14957 | 1870 | |
| 27B | 8192 | 0 | 1 | 8 | 1 | 8 | 64  | 8  | 24.89 | 209.94 | 21062 | 2633 | |
| 27B | 8192 | 4 | 1 | 8 | 1 | 8 | 64  | 8  | 33.45 | 178.00 | 15676 | 1959 | |


Command (4B MIMO):

```bash
export HF_MODEL="Qwen/Qwen3.5-4B"
bash tune_4b_mimo.sh
```

Command (9B MIMO):

```bash
export HF_MODEL="Qwen/Qwen3.5-9B"
bash tune_9b_mimo.sh
```

Command (27B VLM 2nodes):

```bash
export HF_MODEL="Qwen/Qwen3.5-27B"
NNODES=2 NPROC=8 \
  IMAGE_SIZE_W=1920 IMAGE_SIZE_H=1080 \
  NUM_IMAGES=2 SEQ_LEN=4096 MBS=1 GA=8 \
  bash tune_27b_vlm.sh
```


> Multi-node runs: launch the same command on each node with the correct `RANK` and the same `MASTER_ADDR`/`MASTER_PORT`.
> All tuning scripts share the same environment variables. Override them as needed.
> Logs are written to `$OUTPUT_DIR/<tag>/train_node${RANK}.log`.

## Files

| File | Description |
|------|-------------|
| `train_qwen35_vl_mimo.py` | MegatronMIMO training entrypoint. |
| `train_qwen35_vl_vlm.py` | Standard VLM training entrypoint. |
| `tune_4b_mimo.sh` | MFU tuning: Qwen3.5-VL 4B dense (MIMO, 1 node). |
| `tune_9b_mimo.sh` | MFU tuning: Qwen3.5-VL 9B dense (MIMO, 1 node). |
| `tune_27b_vlm.sh` | MFU tuning: Qwen3.5-VL 27B dense (VLM, 2 node). |
| `parse_steady_state.py` | Parse `train_node*.log` and compute steady-state metrics. |
| `export_mfu_results.sh` | One-click export of all tuning results to Excel. |

## 890P Quick Start

Build and run the 890P image from the repository root:

```bash
docker build -f dockers/Dockerfile.890P.Qwen35 -t ppu-qwen35:890P .
docker run --privileged --ipc=host --pid=host --network=host -v /etc/environment:/etc/environment  --ulimit memlock=-1 --shm-size=10.24gb \
  --ulimit stack=67108864 --init -dt --cap-add=IPC_LOCK \
  -v $(pwd):/workspace/ppu_train_recipe \
  --name=qwen35_ppu_train \
  ppu-qwen35:890P /bin/bash
```

## Common Options

| Variable | Meaning |
|----------|---------|
| `HF_MODEL` | Hugging Face model path or local checkpoint. |
| `DATASET_ROOT` | Dataset root directory. |
| `SEQ_LEN` | Sequence length. |
| `NUM_IMAGES` | Number of images per sample (`0` for text-only). |
| `IMAGE_SIZE_W` / `IMAGE_SIZE_H` | Image resolution. |
| `MBS` / `GA` | Micro-batch size and gradient accumulation. |
| `BEST_LANG` / `BEST_IMG` | MIMO parallelism spec for language and image components. |

The scripts also set PPU/PyTorch optimizations such as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NVTE_CPU_OFFLOAD_V1=1`, and `NVTE_USE_CUTLASS_GROUPED_GEMM=1` by default.
