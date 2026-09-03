# Experiment 1B - discovery atlas

This stage evaluates the exact 1,717-cell `(s, alpha, phi)` grid, records `(beta, gamma)`, uses 64 paired seeds, and keeps synthetic correlations, baseline post-LayerNorm activations, causal masks, and unmasked attention as separate strata. It records forward, spectrum, output-similarity, softmax-Jacobian, and one-batch backward metrics. Sign symmetry is checked separately.

Raw resumable records are stored under `data/<preset>/atlas_shard_*.jsonl`. Analysis produces `atlas_summary.csv`, seed-bootstrap intervals, pilot-derived Jacobian/update/rank floors, descriptive labels, and labels recomputed at +/-0.05 probability/entropy thresholds. Quick mode uses a reduced grid and is not proposal evidence.

## Compute

- Preferred machine: multi-GPU node (4-8 recent 24+ GB GPUs) or a GPU cluster. One process per GPU with disjoint shards.
- Kaggle (2x T4/P100): use `run_kaggle.sh` to launch `torchrun` with one disjoint cell stream per GPU.
- Quick smoke test: 5-20 minutes on one GPU; 15-60 minutes on CPU.
- Full sweep: approximately 2-8 GPU-days total, depending on process launch overhead and GPU generation.
- Storage: approximately 5-20 GB JSONL. Convert to Parquet downstream if needed; raw JSONL is intentionally append-safe.

## Parallel with Experiment 1A

You may start the atlas sweep while Experiment 1A is still running. The stages are independent: 1B does not read 1A outputs. Experiment 1A is a **blocking gate for interpretation and confirmatory claims**, not for launching compute. If 1A later fails, archive the 1B raw shards but do not treat labels or boundaries as proposal evidence until the gate passes.

## Run

Linux:

```bash
bash code/exp1b/run_linux.sh quick
CUDA_VISIBLE_DEVICES=0 bash code/exp1b/run_linux.sh full 0 32
```

Windows PowerShell:

```powershell
.\code\exp1b\run_windows.ps1 quick
.\code\exp1b\run_windows.ps1 full 0 32
```

Launch every shard index, then merge and bootstrap:

```bash
python code/exp1b/run.py --preset full --mode analyze --bootstrap-draws 10000
```

### Kaggle (multi-GPU DDP sharding)

Upload the full `code/` tree to `/kaggle/working` (must include `code/common`, `code/exp1b`, and `code/requirements.txt`). Then from any working directory:

```bash
bash /kaggle/working/code/exp1b/run_kaggle.sh full run 0 1
```

The launcher resolves paths from the script location, so you do not need `cd` into a particular folder first. On a 2-GPU kernel it uses `torchrun --nproc_per_node=2` and writes resumable shards to `/kaggle/working/exp1b/data/full/atlas_shard_*.jsonl`. For a manual multi-notebook split across Kaggle sessions, pass different `SHARD_INDEX` / `NUM_SHARDS` arguments (for example `0 16`, `1 16`, ...).

After all shards finish:

```bash
bash code/exp1b/run_kaggle.sh full analyze
```

Full analysis should only be interpreted after Experiment 1A's gate passes.
