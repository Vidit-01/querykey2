# Experiment 1A - mathematical and implementation gate

This stage tests the exact energy, logit-moment, gradient, rank/common-bias, per-head initialization, and checkpoint-survival claims. `full` starts at 20,000 draws per cell and increases draws until both preregistered standard-error limits pass, up to `--max-draws`. Results are resumable JSONL shards under `data/<preset>/`; `gate_summary.json` is the blocking decision.

## Compute

- Preferred machine: CPU cluster node with 32-64 physical cores, 128 GB RAM. A GPU does not materially accelerate the many small float64 QR decompositions.
- Quick smoke test: 4-15 minutes on a modern 8-core CPU.
- Full run: approximately 3-10 CPU-days total; split with `--num-shards` across nodes. Runtime depends strongly on BLAS and CPU generation.
- Storage: below 1 GB.

These are planning estimates, not measured guarantees. The `elapsed_seconds` field records actual time.

## Run

Linux:

```bash
bash code/exp1a/run_linux.sh quick
bash code/exp1a/run_linux.sh full 0 16
```

Windows PowerShell:

```powershell
.\code\exp1a\run_windows.ps1 quick
.\code\exp1a\run_windows.ps1 full 0 16
```

For a 16-shard full run, launch shard indices 0 through 15, then run:

```bash
python code/exp1a/run.py --preset full --mode analyze
```

Do not run downstream confirmatory stages unless `data/full/gate_summary.json` reports `"passed": true`. Quick mode only validates plumbing and cannot satisfy the preregistered Monte Carlo precision.
