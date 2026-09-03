# Experiment 1D - trainability validation

This stage freezes coefficient points before training, then trains the same one-block decoder on associative recall and a fixed TinyStories byte-token subset. Paired seeds reproduce data order and every non-query/key parameter. The primary statistic is early loss decrease per token; traces also record gradients, update-to-weight ratios, clipping, divergence, final loss, and slope.

Full selection enforces 15 diffuse, 15 concentrated, 15 self-locked, 15 screened, 20 boundary/uncertain points, plus four canonical corners. It fails instead of silently shrinking a stratum.

## Compute

- Preferred machine: 8-GPU node with 24+ GB per GPU.
- Quick associative-only smoke test: 5-20 minutes on one GPU.
- Full paired study: approximately 2-10 GPU-days total for 8 seeds and 2,000 steps.
- Storage: 2-20 GB, mostly step traces and the cached TinyStories byte stream.

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

After every full shard:

```bash
python code/exp1d/run.py --preset full --mode analyze
```

Only a confidence interval excluding zero in the preregistered direction passes the atlas-usefulness gate.
