# Initialization Atlas experiment suite

This directory implements every experimental stage in `output/pdf/research_proposal.pdf`:

1. `exp1a`: exact mathematical and integration gate.
2. `exp1b`: 1,717-cell discovery atlas.
3. `exp1c`: adaptive boundary refinement and width/length transfer.
4. `exp1d`: one-block trainability validation on associative recall and TinyStories.
5. `exp2`: 30M-60M real-model pilot, power calculation, confirmation, and transfer tests.

Each folder contains its own Python entry point, compute/time guidance, Linux Bash launcher, and Windows PowerShell launcher. Every entry point writes a manifest, raw append-safe records, failures, timings, and analysis artifacts beneath that experiment's `data/` directory.

## Environment

Python 3.11 or 3.12 is recommended because PyTorch and CUDA wheels are widely available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code/requirements.txt
```

Install the CUDA-specific PyTorch wheel recommended for the target cluster before installing the remaining requirements if the default wheel is unsuitable.

## Reproducibility rules

- Run stages in order. Experiment 1A is a blocking gate.
- Use `quick` only for code validation. It never receives confirmatory labels.
- Freeze and archive `manifest.json`, selected points, treatment files, input hashes, and all raw shards before unblinding downstream outcomes.
- Give each parallel process a distinct shard index.
- Treat wall-clock estimates in READMEs as planning ranges; measured elapsed time is stored in every result.
- Failed runs, OOMs, NaNs, clipping, and resumptions are retained rather than silently excluded.
