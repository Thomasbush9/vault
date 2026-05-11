# Cluster workflow share — 5-min talk notes (2026-05-11)

For the "workflow download" meeting. No slides, just talking. Goal: practical
description of current + anticipated cluster workflow. Skip scientific motivation.


- Pipeline to generate protein structure/function data managed by a webapp so that it can be used by people without experience. It is managed using snakemake to deal with launching jobs and checking deps. Currently given the inputs it fits the resources based on the number of inputs + lenght and it launches array jobs for each chunk. 
- Each model fits into a single gpu, although for some inputs we reach a OOM -> but there are options 
- Sharing: both kempner specific version and soon container based one 
- Inference based. 
- As for the next steps we are going to build a similar pipeline but for protein design + agents

## 30-second framing

ProtForge = ML pipeline orchestrator for protein structure/function. Today it's
MSA + structure prediction + embeddings; in the next 3–6 months it grows a
**protein-design** front-end (generative models) feeding the same scoring stack.
All on Kempner, SLURM, single-GPU jobs, embarrassingly parallel.

## Current workflow (what runs today)

**Stages.** FASTA → MSA (ColabFold + MMseqs2, CPU) → Boltz (structure, GPU) →
ESM2/3 (embeddings, GPU) and/or ESMFold (alt structure, GPU) → PDAnalysis (CPU).

**Driver.** Snakemake DAG, `snakemake-executor-plugin-slurm` submits each chunk
as its own `sbatch`. Sentinel files (`.boltz_complete` etc.) gate stages. No
multi-node MPI except for the CPU PDAnalysis step.

**Chunking.** Inputs split by file count into chunks (`max_files_per_job: 25`
prod, `1` for calibration). Moving to **bin-aware chunking** — quantile-binned
by sequence length so a chunk of 200aa sequences doesn't get sized for an 1800aa
worst case. Plan in `bin-aware-chunking.md`.

**Sizing.** A small `webapp/estimator.py` predicts wall time + GPU mem from
sequence length per stage / per GPU type, using coefficients calibrated from
real cluster sweeps (`scripts/calibrate/`). Currently H100 calibrated; A100 next.

## Answers to the prompts

- **What's running now / soon at scale?** 7k-FASTA batches through MSA + Boltz
  + ESM + ESMFold on `kempner_h100`. Calibration sweeps (~20 stratified
  sequences) before each big batch. Concurrency cap ~10–20 SLURM jobs.

- **Code location.** GitHub: `protforge` repo. Snakefile + `workflow/rules/*.smk`
  on `snakemake` branch. Bash fallback on `main`. Webapp UI (`webapp/`) generates
  configs + estimates resources before submission.

- **Sharing + records.**
  - Code → GitHub.
  - Cluster data → `/n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/...`,
    local mirror via rsync to a Raspberry Pi store (`~/tmp_data/tmp_data/`).
  - Logs → Obsidian vault `Lab/protforge/` (per-sweep markdown in `log/`,
    durable workflow in `calibration.md`).
  - Benchmarks → Snakemake `benchmark:` TSVs joined with `chunk_stats.tsv`.

- **Model sizes / parallelization.**
  - Boltz ~1B, ESM2 up to 15B (we use 3B/8B), ESMFold ~650M.
  - **Data-parallel only.** Each sequence/chunk is its own SLURM job, one GPU.
    No tensor/pipeline parallelism, no NCCL, no multi-node compute.

- **Multi-node?** Only for the CPU MPI PDAnalysis stage. Sequence stages are
  single-node single-GPU.

- **Fit on 1 GPU?** Yes for typical L < 1500. ESMFold OOMs above ~1800aa on
  H100 80GB → fixed by per-seq trunk chunking (`--chunk_size_threshold 1200`).
  Boltz has its own long-protein failure mode (silent organize failure on
  ≥1801aa) still being worked on.

- **Memory / comms issues.** Memory cliffs on long sequences (O(L²) attention).
  No communication issues — no inter-process comms by design. SLURM `cgroups`
  accounting bites on MSA because the host RSS looks like 140 GB but it's
  mmap'd MMseqs2 DB (real allocation ~48 GB).

- **Inference on large models?** Yes — Boltz, ESM2/3, ESMFold are all
  inference-only here. Run via Boltz's CLI, fair-esm Python API, HuggingFace
  for ESMFold. Standard PyTorch, no custom kernels.

- **Standard open models used.** Boltz1, ESM2 (3B/8B), ESM3 small, ESMFold v1.
  Plus MMseqs2 + ColabFold DBs. All from the shared lab cache at
  `/n/holylfs06/.../protforge/`.

## Next 3–6 months — protein design pipeline

Same orchestration shape (Snakemake + SLURM array), new front-end stages:

**New generative stages (planned).**
1. **Backbone generation** — RFdiffusion / Chroma (diffusion over coordinates,
   100s of denoising steps per sample). Inference, ~1 GPU per sample, minutes
   to hours per design depending on target length.
2. **Sequence design** — ProteinMPNN or ESM3-IF (inverse folding). Fast
   inference, single GPU, batched.
3. **Validation** — feed designs back into existing Boltz / ESMFold / ESM stack
   for ranking. *This reuses the entire current pipeline unchanged.*

**Anticipated cluster differences from today.**
- **Many more independent jobs** (1 design = 1 GPU job, but we want 10³–10⁴
  designs per target). Snakemake DAG width grows; may need a higher
  concurrency cap or job arrays instead of one job per design.
- **Still single-GPU, still data-parallel.** No current need for multi-node
  training or tensor parallel — these are inference-only models that fit on
  one H100/A100.
- **Possible exception:** ESM3 large (98B params) for in-context design might
  need FSDP across 2× H100. Not committed yet.
- Storage shifts to *many small outputs* per design (PDB + sequence + score)
  rather than fewer large MSA files. Will need to revisit the per-batch output
  layout and rsync strategy.
- Calibration story carries over directly — same `calibrate.sh` workflow,
  new stages added to `scaling_models.yaml`.

**What we're NOT planning (yet).**
- Training/fine-tuning runs. No DDP, no multi-node, no checkpointing pipeline.
- Custom CUDA. Sticking with off-the-shelf model code.
- Non-SLURM execution (Singularity container is on the longer list for
  portability but not 3–6 mo critical).

## Open questions / things to ask the room

- Anyone doing multi-node inference for big LMs? FSDP setup on Kempner?
- GPU memory accounting — how are others capturing real GPU peak vs host RSS?
  (We're polling `nvidia-smi`; curious if anyone has a cleaner hook.)
- Standard pattern for 10k+ tiny-output jobs without melting the FS metadata?
