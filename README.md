# LIMBER — Layer-Interleaved Mamba–Bidirectional Encoder for RNA

LIMBER is a fully open non-coding RNA foundation model that interleaves bidirectional
self-attention with selective state-space (Mamba) layers in a repeating
**Transformer–Transformer–Mamba (TTM)** pattern. It is pretrained with masked language
modelling on ~40.5M RNAcentral sequences.

This repository holds the pretraining code, a task-agnostic evaluation and interpretation
framework, and the downstream pipelines used to compare LIMBER against published RNA
language models under matched conditions.

Alongside LIMBER we pretrained size-matched **pure Transformer** and **pure Mamba** models
on the same data and optimizer settings, so architecture can be separated from model, data
and compute scale.

---

## Model zoo

Weights live in a single Hugging Face repo, one subfolder per model:
[`Ziyao1010/LIMBER`](https://huggingface.co/Ziyao1010/LIMBER).

| Model | Layers | Hidden | Heads | FFN | Pattern | Params |
|---|---|---|---|---|---|---|
| LIMBER-10M | 12 | 256 | 4 | 1024 | TTM × 4 | 10.2M |
| LIMBER-50M | 15 | 512 | 8 | 2048 | TTM × 5 | 49.3M |
| LIMBER-nano (100M) | 18 | 640 | 10 | 2560 | TTM × 6 | 91.7M |
| LIMBER-meso (300M) | 15 | 1280 | 20 | 5120 | TTM × 5 | 303.6M |
| LIMBER-macro (650M) | 33 | 1280 | 20 | 5120 | TTM × 11 | 664.3M |
| Transformer-100M / 300M / 650M | 18 / 15 / 33 | 640 / 1280 / 1280 | 10 / 20 / 20 | 2560 / 5120 / 5120 | all T | 89.7M / 298.1M / 652.3M |
| Mamba-100M / 300M / 650M | 18 / 15 / 33 | 640 / 1280 / 1280 | — | — | all M | 95.8M / 314.4M / 688.2M |

Mamba layers use a 16-dimensional state space, causal convolution width 4 and expansion
factor 4. Note that the Mamba layers are **causal**, so a pure-Mamba model sees only
upstream context; the Transformer layers in LIMBER are bidirectional.

Additional checkpoints in the same HF repo cover the scaling studies: LIMBER-nano trained on
20/40/60/80/100% of the corpus, a `d_conv = 2` convolution-width ablation, and per-epoch
checkpoints for the training-budget analysis.

---

## Quick start

```bash
pip install torch transformers safetensors
pip install causal-conv1d mamba-ssm      # required for the Mamba layers
```

Download a checkpoint and run a forward pass:

```bash
python scripts/load_hybrid650.py                 # or --model_dir <local> / --seq ACGU...
```

```python
from scripts.load_hybrid650 import load_hybrid, encode
import torch

model, _ = load_hybrid()                         # HybridMambaForMaskedLM, eval mode
ids = torch.tensor([encode("GGGCUAUUAGCUCAGUUGG")]).cuda()
out = model(input_ids=ids, attention_mask=torch.ones_like(ids))   # out.logits: [1, L+2, 10]
```

To load any checkpoint directly:

```python
import json, torch
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM

cfg = HybridMambaConfig(**json.load(open(f"{ckpt_dir}/model_config.json")))
model = HybridMambaForMaskedLM(cfg)
model.load_state_dict(torch.load(f"{ckpt_dir}/pytorch_model.bin", weights_only=True))
```

The same class covers all three architectures — `layer_pattern` in `model_config.json`
selects them (`TTMTTM...`, `TTTTTT...` or `MMMMMM...`).

**Tokenization.** Single-nucleotide resolution, 10 tokens: `[PAD] [UNK] [CLS] [SEP] [MASK]`
plus `A U C G N` (`tokenizers/single/vocab.txt`). Preprocessing uppercases the sequence,
maps T to U, and maps any non-ACGU symbol to N.

---

## Pretraining

```bash
python scripts/train_hybrid650_mlm.py --help      # single entry point for all three architectures
bash   scripts/run_hybrid650.sh                   # Slurm wrapper
```

- **Corpus.** RNAcentral, exact-sequence deduplicated to **40,498,819** sequences plus a
  fixed held-out validation set. Sequences ≤ 1,022 nt are used whole; longer ones are
  randomly cropped to 1,022 nt, resampled each epoch. Streaming loader with a 50k shuffle
  buffer: `scripts/rna_stream_dataset.py`.
- **Objective.** BERT-style MLM: 15% of nucleotide positions selected, of which 80% become
  `[MASK]`, 10% a random canonical nucleotide and 10% are left unchanged. Cross-entropy is
  computed only over the selected positions.
- **Optimization.** AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8), weight decay 0.01, peak learning rate
  5e-5 with 2,000-step linear warmup then cosine decay to 1e-5, gradient clipping 1.0,
  bfloat16 mixed precision, dropout 0.1, seed 42. Effective batch 2,688 sequences, so one
  pass over the corpus is 30,134 optimizer steps. Pretrained on 4× NVIDIA H200.

**Perplexity** is a masked-LM perplexity: the exponential of the mean negative
log-likelihood over masked positions only, on a fixed 4,096-sequence validation subset with
masks drawn from a fixed seed so every model is scored at identical positions
(`scripts/baseline_ppl_only.py`, `scripts/layer_importance.py`).

---

## Task-agnostic evaluation and interpretation

These probes need no labelled training and no fine-tuning, so they measure what pretraining
itself produced.

| Probe | What it measures | Script |
|---|---|---|
| Categorical Jacobian | Secondary-structure contacts from the sensitivity of output logits to every single-nucleotide substitution, scored by normalized AUPR | `scripts/jacobian_ss_eval.py`, `scripts/extract_contacts_categorical_jacobian.py` |
| Embedding perturbation | The same contacts, but from layer-wise hidden-state responses, giving one contact map per layer | `scripts/extract_contacts_embedding_perturb.py` |
| Family / type classification | Overlap between similarity distributions of homologous and non-homologous sequence pairs, in FFT spectral features of each layer | `scripts/extract_family_cls_full.py`, `scripts/nrc_zeroshot.py` |
| Zero-shot fitness | Spearman correlation between measured mutational effects and wild-type-conditioned log-likelihood ratios | scoring scripts are external to this repo; figure: `scripts/make_fig_fitness_landscape.py` |
| Layer importance | Change in perplexity, contact recovery and fitness correlation when a single layer is ablated or the encoder is truncated | `scripts/layer_importance.py` |

Evaluation data: bpRNA-1m(90) for structure (7 source datasets, 26,707 sequences after
removing > 90% identity and > 512 nt), Rfam 15.1 seed alignments and ArchiveII for
classification, and deep mutational scanning assays for fitness.

---

## Downstream tasks

Every task keeps the published pipeline intact and swaps only the RNA encoder, so the
encoder is the single variable. Baselines are RiNALMo-giga, AIDO.RNA-650M, RNA-FM and
StructRFM, each run through the same protocol as LIMBER.

| Task | Data | Entry point |
|---|---|---|
| Secondary structure (intra-family) | SPOT-RNA bpRNA splits TR0/VL0/TS0; Rivas TrainSetA → TestSetA / TestSetB | `scripts/ss_bench.sbatch` |
| Secondary structure (inter-family) | ArchiveII leave-one-family-out, 9 families | `scripts/rinalmo_ss_finetune.py`, `scripts/ss_perseq_external.py` |
| Tertiary structure | trRosettaRNA / Zfold on CASP15-natural, CASP16 and RNA-Puzzles | `rna3d/` (driver scripts; the vendored Zfold source is not tracked here) |
| ncRNA classification | nRC, 13 classes | `scripts/nrc_finetune.py` (fine-tune), `scripts/nrc_zeroshot.py` (kNN probe) |
| IRES identification | DeepIRES benchmark, derived from the Weingarten-Gabbay 55k oligo screen | `scripts/downstream_finetune.py --task ires` |
| Mean ribosome load | 5′UTR MPRA library | `scripts/downstream_finetune.py --task mrl` |
| RNA–small-molecule interaction | SMRTnet dataset and its five external benchmark sets | see `scripts/` and the SMRTnet integration notes |

Secondary-structure fine-tuning follows RiNALMo's published protocol: a residual
convolutional pairwise head, BCE on the upper triangle, gradual unfreezing of three encoder
blocks every three epochs, Adam at 5e-4 with newly unfrozen groups at a tenth of that,
LinearLR 1.0 → 0.1 over 7,000 steps, a validation-tuned decision threshold searched over
0.01–0.29, and flexible (±1 nucleotide) F1 with a canonical-pair greedy decoder. The
`.ct` / `.st` / `.bpseq` parsers and metrics in `scripts/rinalmo_ss_lib.py` were checked
against RiNALMo's reference implementation and agree exactly.

---

## Repository layout

```
models/              HybridMambaForMaskedLM — all three architectures
scripts/             pretraining, evaluation probes, downstream pipelines, figures
rna3d/               tertiary-structure driver and scoring scripts
tokenizers/single/   single-nucleotide vocabulary
data/                small metadata only
HANDOFF.md           historical notes from the first pretraining run
```

Checkpoints, run outputs, logs and benchmark corpora are **not** tracked here — they are
regenerable or too large, and live on the project filesystem. `scripts/upload_to_hf.py`
pushes a finished checkpoint to the Hugging Face repo.

---

## Notes and limitations

- Mamba layers are causal. A pure-Mamba encoder therefore cannot see downstream context,
  which matters for any task that depends on base pairing; this is a property of the
  architecture comparison, not a bug.
- Position embeddings cover 1,026 tokens, so the usable sequence length is 1,022 nt. Longer
  sequences must be cropped or windowed.
- Published numbers are produced with several random seeds where the protocol allows it;
  differences smaller than the seed-to-seed spread should not be read as real.

## License

Not yet specified.
