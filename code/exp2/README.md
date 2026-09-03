# Experiment 2 - controlled real-language-model study

This stage trains an ordinary decoder-only Transformer under the six proposal treatments. The full model has 8 layers, 8 heads, width 576, and approximately 32M parameters. It uses byte tokens to keep tokenizer identity fixed and transparent. Query/key tensors are separate `[output, input]` linear projections; per-head initialization, reshape order, manual logits, post-mask attention, and bitwise checkpoint reload are checked before training.

Treatments 2-5 share the frozen atlas scale. The diverse and layer-homogeneous controls have an exactly matched model-wide coefficient histogram when the full 8-point portfolio is used. Head/layer permutations are seed-fixed and saved. The orthogonal endpoint is included only if its Stage 1D runs pass.

## Compute

- Preferred machine: 8x A100/H100 80 GB nodes or an equivalent cluster. The script runs one treatment/seed job per GPU process; use shards to distribute jobs.
- Quick end-to-end smoke test: 10-40 minutes on one GPU.
- External 2M-token pilot: approximately 1-4 GPU-hours per arm/seed.
- 100M-token confirmation: approximately 1-4 GPU-days per arm/seed, or roughly 80-600 A100/H100 GPU-days for all treatments and the power-determined seed count.
- Storage: 200-800 GB for checkpoints, optimizer state, traces, cached data, and raw metrics. Remove no failed-run evidence; archive checkpoints after analysis.

Times are intentionally broad estimates. Actual token throughput and elapsed seconds are recorded.

## Data

For immutable runs, pass local TinyStories JSONL/text files:

```text
--train-file /data/tinystories_train.jsonl
--validation-file /data/tinystories_validation.jsonl
```

If omitted in full mode, the Hugging Face TinyStories train/validation splits are downloaded once and converted to saved uint8 streams. Quick mode uses a clearly marked plumbing-only toy stream.

## Required sequence

1. Freeze treatments and pass integration checks.
2. Run an external pilot with at least three paired seeds.
3. Analyze the pilot and freeze the 80%-power seed count.
4. If feasible, run confirmation with at least eight paired seeds and 100M tokens.
5. Analyze the two paired primary contrasts with Holm correction and the +/-0.01 nat equivalence margin.
6. Freeze the winning treatment and run `--study transfer` with `--variant second_width`, `second_context`, `deeper`, and `second_corpus`. For `second_corpus`, pass alternate train/validation files.

Linux:

```bash
bash code/exp2/run_linux.sh quick pilot
bash code/exp2/run_linux.sh full pilot 0 32 /data/train.jsonl /data/validation.jsonl
python code/exp2/run.py --preset full --study pilot --mode analyze
bash code/exp2/run_linux.sh full confirm 0 64 /data/train.jsonl /data/validation.jsonl
```

Windows PowerShell:

```powershell
.\code\exp2\run_windows.ps1 quick pilot
.\code\exp2\run_windows.ps1 full pilot 0 32 C:\data\train.jsonl C:\data\validation.jsonl
python code/exp2/run.py --preset full --study pilot --mode analyze
.\code\exp2\run_windows.ps1 full confirm 0 64 C:\data\train.jsonl C:\data\validation.jsonl
```

Confirmation aborts if the pilot power calculation exceeds `--max-seeds`; it does not silently cap an underpowered study.

Example transfer test:

```bash
python code/exp2/run.py --preset full --study transfer --variant second_width --mode run \
  --train-file /data/train.jsonl --validation-file /data/validation.jsonl
```
