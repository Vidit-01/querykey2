# Experiment 1B - discovery atlas

This stage evaluates the exact 1,717-cell `(s, alpha, phi)` grid, records `(beta, gamma)`, uses 64 paired seeds, and keeps synthetic correlations, baseline post-LayerNorm activations, causal masks, and unmasked attention as separate strata. It records forward, spectrum, output-similarity, softmax-Jacobian, and one-batch backward metrics. Sign symmetry is checked separately.

Raw resumable records are stored under `data/<preset>/atlas_shard_*.jsonl`. Analysis produces `atlas_summary.csv`, seed-bootstrap intervals, pilot-derived Jacobian/update/rank floors, descriptive labels, and labels recomputed at +/-0.05 probability/entropy thresholds. Quick mode uses a reduced grid and is not proposal evidence.

## Compute

- Preferred machine: multi-GPU node (4-8 recent 24+ GB GPUs) or a GPU cluster. One process per GPU with disjoint shards.
- Quick smoke test: 5-20 minutes on one GPU; 15-60 minutes on CPU.
- Full sweep: approximately 2-8 GPU-days total, depending on process launch overhead and GPU generation.
- Storage: approximately 5-20 GB JSONL. Convert to Parquet downstream if needed; raw JSONL is intentionally append-safe.

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

Full analysis should only be interpreted after Experiment 1A's gate passes.
