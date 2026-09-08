# Experiment 1D - trainability validation

This stage freezes coefficient points before training, then trains the same one-block decoder on associative recall and a fixed TinyStories byte-token subset. Paired seeds reproduce data order and every non-query/key parameter. The primary statistic is early loss decrease per token; traces also record gradients, update-to-weight ratios, clipping, divergence, final loss, and slope.

Full selection enforces 15 diffuse, 15 concentrated, 15 self-locked, 15 screened, 20 boundary/uncertain points, plus four canonical corners. It fails instead of silently shrinking a stratum. Adaptive 1C cells are labeled by majority vote on the reference geometry; concentrated and self-locked cells, which 1C did not sample, are filled from the 1B discovery atlas (`--discovery-summary`).

## Compute

- Preferred machine: 8-GPU node with 24+ GB per GPU.
- Quick associative-only smoke test: 5-20 minutes on one GPU.
- Full paired study: approximately 2-10 GPU-days total for 8 seeds and 2,000 steps.
- Storage: 2-20 GB, mostly step traces and the cached TinyStories byte stream.

Each training job is an independent `(point, task, seed)` run of a tiny one-block model. That job cannot fill a 15 GB GPU by itself (~0.5 GB, often ~30-40% SM util). Launchers therefore pack several workers per GPU (`torchrun --nproc_per_node=<gpus * workers>`), not data-parallel one model. Default is 8 workers per GPU (`WORKERS_PER_GPU=8`). Lower it if you OOM; raise it if utilization is still low.

## Data

Pass `--tinystories-file path/to/train.jsonl` for an immutable local subset. Without it, full mode downloads `roneneldan/TinyStories` through Hugging Face and saves the exact uint8 cache under `data/`. Quick launchers skip text by default; add `--include-text` to exercise that path.

## Run

Linux:

```bash
bash code/exp1d/run_linux.sh quick
CUDA_VISIBLE_DEVICES=0 bash code/exp1d/run_linux.sh full 0 16 /data/TinyStories-train.jsonl
```

Windows PowerShell:

```powershell
.\code\exp1d\run_windows.ps1 quick
.\code\exp1d\run_windows.ps1 full 0 16 C:\data\TinyStories-train.jsonl
```

If CUDA is visible, the launchers start `torchrun` with `GPUs × 8` processes so each device runs several trainings at once. Pin a single GPU with `CUDA_VISIBLE_DEVICES=0` when you want one device; it will still pack 8 workers on that GPU.

After every full shard:

```bash
python code/exp1d/run.py --preset full --mode analyze
```

### Kaggle (multi-GPU sharding)

Upload the full `code/` tree to `/kaggle/working` (`code/common`, `code/exp1d`, and `code/requirements.txt`). Attach a TinyStories jsonl if you do not want Hugging Face downloads.

```bash
# Freeze points once (optional if bundled selected_points.json is present)
bash /kaggle/working/code/exp1d/run_kaggle.sh full select

# Run shard 0 of 16 (2-GPU kernel packs 8 workers per GPU by default)
bash /kaggle/working/code/exp1d/run_kaggle.sh full run 0 16

# After every shard finishes
bash /kaggle/working/code/exp1d/run_kaggle.sh full analyze
```

Optional environment variables:

- `KAGGLE_TINYSTORIES` — path to a local TinyStories jsonl
- `KAGGLE_OUTPUT` — defaults to `/kaggle/working/exp1d/data/<preset>`
- `BOOTSTRAP_DRAWS` — defaults to `10000` on analyze
- `WORKERS_PER_GPU` — independent trainings packed on each GPU (default 8)

On a 2-GPU kernel, `run` mode launches `torchrun --nproc_per_node=16` (8 workers × 2 GPUs) and maps ranks onto `cuda:0` / `cuda:1`. Rank 0 builds the TinyStories cache and writes selection. For multi-notebook splits, pass different `SHARD_INDEX` / `NUM_SHARDS` pairs (`0 16`, `1 16`, ...).

Only a confidence interval excluding zero in the preregistered direction passes the atlas-usefulness gate.
