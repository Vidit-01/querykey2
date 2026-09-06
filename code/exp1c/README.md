# Experiment 1C - adaptive refinement and transfer

This stage fits a probabilistic visualization surrogate on discovery-only labels, holds out 20% of coefficient cells, reports calibration/balanced accuracy/boundary error, selects uncertain and theory-disagreement points, measures adaptive cells, and repeats the map at transfer configurations.

**Default full preset (lighter):** 120 adaptive cells, 64 seeds, geometries $(64,16,64)$ and $(128,16,128)$ only.

**Preregistered full replication:** pass `--adaptive-cells 500 --seeds 256 --replicate-discovery-grid` to include all four transfer geometries and the original 1,717 discovery cells.

The frozen `candidate_portfolio.json` uses replicated screened points without seeing training outcomes.

## Compute

- Preferred machine: GPU cluster with 8-32 recent GPUs and 256+ GB aggregate host RAM.
- Quick smoke test: 10-30 minutes on one GPU after Experiment 1B quick mode.
- Default full adaptive run: roughly 2-8 GPU-days total.
- Preregistered replication (`--replicate-discovery-grid`): roughly 60-200 GPU-days total.
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

### Kaggle (multi-GPU DDP sharding)

You only need **1B `atlas_summary.csv`** (not the 1.3 GB raw shards). Attach it as a Kaggle dataset or copy it to `/kaggle/working/exp1b/data/full/atlas_summary.csv`.

Upload the full `code/` tree to `/kaggle/working` (`code/common`, `code/exp1b`, `code/exp1c`, `code/requirements.txt`).

```bash
# Once per session: fit surrogate + write adaptive_points.json
bash /kaggle/working/code/exp1c/run_kaggle.sh full propose

# Run shard 0 of 16 (2-GPU kernel uses torchrun automatically)
bash /kaggle/working/code/exp1c/run_kaggle.sh full run 0 16

# After every shard finishes
bash /kaggle/working/code/exp1c/run_kaggle.sh full analyze
```

Optional environment variables:

- `KAGGLE_DISCOVERY_SUMMARY` — path to 1B `atlas_summary.csv`
- `KAGGLE_OUTPUT` — defaults to `/kaggle/working/exp1c/data/<preset>`
- `BOOTSTRAP_DRAWS` — defaults to `10000` on analyze

On a 2-GPU kernel, `run` mode launches `torchrun --nproc_per_node=2` and splits tasks across GPUs. For multi-notebook splits, pass different `SHARD_INDEX` / `NUM_SHARDS` pairs (`0 16`, `1 16`, ...).

The surrogate interpolates the observed map; it is not itself evidence.
