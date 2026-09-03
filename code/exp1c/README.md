# Experiment 1C - adaptive refinement and transfer

This stage fits a probabilistic visualization surrogate on discovery-only labels, holds out 20% of coefficient cells, reports calibration/balanced accuracy/boundary error, selects uncertain and theory-disagreement points, measures at least 500 new cells with 256 seeds, and repeats the map at the three transfer configurations.

`--replicate-discovery-grid` includes the original 1,717 cells alongside adaptive cells at every transfer width. The launcher enables it for full proposal runs. The frozen `candidate_portfolio.json` uses replicated screened points without seeing training outcomes.

## Compute

- Preferred machine: GPU cluster with 8-32 recent GPUs and 256+ GB aggregate host RAM.
- Quick smoke test: 10-30 minutes on one GPU after Experiment 1B quick mode.
- Full adaptive cells only: roughly 10-35 GPU-days total.
- Full proposal replication including the discovery grid: roughly 60-200 GPU-days total.
- Storage: approximately 50-250 GB for raw JSONL, depending on replication scope.

The large range reflects hardware and JSON serialization overhead. Use many disjoint shards and preserve all shard files.

## Run

Linux:

```bash
bash code/exp1c/run_linux.sh quick
CUDA_VISIBLE_DEVICES=0 bash code/exp1c/run_linux.sh full 0 128
```

Windows PowerShell:

```powershell
.\code\exp1c\run_windows.ps1 quick
.\code\exp1c\run_windows.ps1 full 0 128
```

After all full shards:

```bash
python code/exp1c/run.py --preset full --mode analyze --bootstrap-draws 10000
```

The surrogate interpolates the observed map; it is not itself evidence.
