---
tags:
  - rna
  - biology
  - foundation-model
  - rna-language-model
  - mamba
  - state-space-model
  - masked-language-modeling
  - rna-structure
---

<div align="center">

<img src="https://raw.githubusercontent.com/ziyaow1010/LIMBER/main/assets/limber_banner_light.png" alt="LIMBER — Layer-Interleaved Mamba-Bidirectional Encoder for RNA" width="100%">

<h3>A Hybrid RNA Foundation Model for Predicting and Understanding<br>RNA Structure and Function</h3>

<!-- TODO: add the bioRxiv DOI once the preprint is live -->
[**Paper**](#) · [**Code**](https://github.com/ziyaow1010/LIMBER) · [**Quick start**](#quick-start)

**Ziyao Wang**<sup>†</sup> · **Yunrui Qiu**<sup>†*</sup> · Yexiao He · Isabel Thompson · Hongli Wang · Adam Porter · **Ang Li**<sup>*</sup>

<sub>University of Maryland, College Park · University of Notre Dame<br><sup>†</sup>Equal contribution · <sup>*</sup>Corresponding authors</sub>

</div>

---

**LIMBER** (Layer-Interleaved Mamba–Bidirectional Encoder for RNA) is a hybrid RNA foundation
model that interleaves bidirectional Transformer attention with Mamba selective state-space
layers in a repeating **Transformer–Transformer–Mamba (TTM)** pattern, pretrained by masked
language modeling on **40,498,819 RNAcentral sequences**.

Self-attention captures direct couplings between arbitrary positions — the long-range
coevolutionary signal that base pairing leaves in a sequence. Mamba contributes a complementary
prior for ordered structure: RNA is a directional polymer, and its recurring motifs carry ordered
signatures that attention alone does not impose. Interleaving the two gives the model both.

This repository holds **every checkpoint from the paper**, one subfolder per model: the three
headline LIMBER models, the size- and data-scaling panel, and the size-matched pure-Transformer
and pure-Mamba controls trained under identical settings.

## What's in this repo

### Main models

| Subfolder | Layers | Hidden | Heads | FFN | Params |
|---|---|---|---|---|---|
| **`LIMBER-macro`** | 33 | 1280 | 20 | 5120 | 664.3M |
| **`LIMBER-meso`** | 15 | 1280 | 20 | 5120 | 303.6M |
| **`LIMBER-nano`** | 18 | 640 | 10 | 2560 | 91.7M |

`LIMBER-macro` is the model reported throughout the paper. Start there unless you are
compute-constrained — `LIMBER-nano` matches or beats models six times its size on several tasks.

### Controlled architecture comparison

Pure-Transformer and pure-Mamba encoders at matched width, depth, data and optimizer settings, so
the architecture is the only variable: `bert_100m_epoch6`, `bert_300m_epoch6`, `bert_650m_epoch6`,
`kmer1-PureMamba-100M`, `kmer1-PureMamba-300M`, `PureMamba-650M`.

### Scaling and ablation panel

| Axis | Subfolders |
|---|---|
| Model size | `LIMBER-10M`, `LIMBER-50M` |
| Pretraining-data fraction | `LIMBER-nano-data20pct`, `-data40pct`, `-data60pct`, `-data80pct` |
| Training budget | `LIMBER-nano-ep6`, `LIMBER-meso-ep6`, `LIMBER-macro-ep1`, `-ep6`, `-ep9` |
| Shape at fixed budget | `LIMBER-nano-thin`, `LIMBER-nano-wide`, `LIMBER-meso-thin`, `LIMBER-6L256-early` |
| Convolution width | `LIMBER-nano-dconv2` (width 2 instead of 4) |
| Tokenization and objective | `kmer1-BERT`, `kmer6-BERT`, `kmer6-Hybrid`, `kmer1-decoder`, `kmer6-decoder` |

## Quick start

```bash
pip install torch transformers safetensors huggingface_hub
pip install causal-conv1d mamba-ssm      # required for the Mamba layers
git clone https://github.com/ziyaow1010/LIMBER && cd LIMBER
```

```bash
python scripts/load_hybrid650.py        # downloads LIMBER-macro and runs a forward pass
```

```python
import torch
from scripts.load_hybrid650 import load_hybrid, encode, embed

model, _ = load_hybrid()                                  # HybridMambaForMaskedLM, eval mode
ids = torch.tensor([encode("GGGCUAUUAGCUCAGUUGG")]).cuda()

logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
h      = embed(model, ids)                                # final-layer representations

logits.shape     # [1, L+2, 10]    masked-LM logits over the nucleotide vocabulary
h.shape          # [1, L+2, 1280]  per-nucleotide embeddings for downstream heads
h[:, 0]          # [1, 1280]       [CLS] — the pooled sequence representation
```

Load any other subfolder directly:

```python
import json, torch
from huggingface_hub import snapshot_download
from models.hybrid_mamba_bert import HybridMambaConfig, HybridMambaForMaskedLM

d = snapshot_download("Ziyao1010/LIMBER", allow_patterns=["LIMBER-nano/*"]) + "/LIMBER-nano"
model = HybridMambaForMaskedLM(HybridMambaConfig(**json.load(open(f"{d}/model_config.json"))))
model.load_state_dict(torch.load(f"{d}/pytorch_model.bin", weights_only=True))
```

One class, `HybridMambaForMaskedLM`, covers all three architectures — `layer_pattern` in
`model_config.json` selects `TTMTTM...`, `TTTTTT...` or `MMMMMM...`.

**Tokenization.** Single-nucleotide resolution, 10 tokens: `[PAD] [UNK] [CLS] [SEP] [MASK]` plus
`A U C G N`. Preprocessing uppercases the sequence, maps T to U, and maps any non-ACGU symbol
to N. Position embeddings cover 1,026 tokens, giving a usable sequence length of 1,022 nt.

## Results

<img src="https://raw.githubusercontent.com/ziyaow1010/LIMBER/main/assets/limber_benchmarks_light.png" alt="LIMBER benchmark results" width="100%">

Every baseline is fine-tuned and evaluated by us under the same protocol as LIMBER, so the RNA
encoder is the only variable.

**RNA secondary structure** — flexible F1 (±1 nt), RiNALMo fine-tuning protocol.

| Model | bpRNA TS0 (median) | Rivas TestSetB (median) | ArchiveII inter-family (mean) |
|---|---|---|---|
| **LIMBER-macro** | **0.895** | **0.816** | 0.809 |
| **LIMBER-meso** | 0.889 | 0.791 | 0.773 |
| **LIMBER-nano** | 0.870 | 0.772 | 0.736 |
| AIDO.RNA-650M | 0.851 | 0.810 | **0.819** |
| RiNALMo-giga | 0.851 | 0.720 | 0.789 |
| structRFM | 0.792 | 0.613 | 0.659 |
| RNA-FM | 0.702 | 0.492 | 0.401 |

**ncRNA classification** — nRC, 13 classes, 2,600 test sequences.

| Model | Accuracy | MCC | Macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| **LIMBER-macro** | **0.9919** | **0.9913** | **0.9919** | **0.9996** | **0.9982** |
| **LIMBER-meso** | 0.9915 | 0.9908 | 0.9916 | 0.9989 | 0.9943 |
| **LIMBER-nano** | 0.9865 | 0.9854 | 0.9866 | 0.9988 | 0.9933 |
| **LIMBER-macro, first 14 layers** | 0.9842 | 0.9829 | 0.9842 | 0.9995 | 0.9968 |
| AIDO.RNA-650M | 0.9865 | 0.9854 | 0.9865 | 0.9994 | 0.9964 |
| RiNALMo-giga | 0.9792 | 0.9775 | 0.9792 | 0.9991 | 0.9939 |
| RNA-FM | 0.9650 | 0.9621 | 0.9648 | 0.9985 | 0.9905 |
| structRFM | 0.9638 | 0.9609 | 0.9638 | 0.9970 | 0.9851 |

Truncating LIMBER-macro to its first 14 layers keeps ~43% of the parameters and still beats
RiNALMo-giga — a prediction that came out of the layer-wise analysis before any fine-tuning was run.

**Mean ribosome load** — 5′UTR MPRA, R²: LIMBER-macro 0.891 (random) / 0.826 (human);
LIMBER-nano 0.882 / 0.819, far ahead of the other ~100M models.

**IRES identification** — on the DeepIRES benchmark LIMBER-macro reaches 0.899 accuracy, the
highest of any model tested, and takes the top AUPRC (0.852) on the independent 104-sequence
benchmark, ahead of the task-specific DeepIRES.

**RNA–small-molecule interactions** — dropped into the SMRTnet multimodal framework, LIMBER-macro
surpasses RiNALMo-giga and AIDO.RNA-650M, and LIMBER-nano outperforms RNA-FM, structRFM and
SMRTnet's own RNASwan-seq encoder.

**Zero-shot, no fine-tuning** — across 26,707 bpRNA-1m(90) sequences, categorical-Jacobian contact
maps read straight out of LIMBER-macro recover secondary structure better than RNA-FM, structRFM,
RNAErnie and the LucaOne family across nearly every RNA category, and beat RNAstructure and
ViennaRNA in most. On 33 deep mutational scanning assays from RNAGym (862,376 variants),
wild-type-conditioned log-likelihood ratios give the leading overall Spearman correlation.

## Training details

- **Corpus.** RNAcentral, exact-sequence deduplicated to 40,498,819 sequences plus a fixed
  held-out validation set. Sequences ≤ 1,022 nt used whole; longer ones randomly cropped to
  1,022 nt, resampled each epoch.
- **Objective.** BERT-style MLM: 15% of positions selected, of which 80% → `[MASK]`, 10% → a
  random canonical nucleotide, 10% unchanged. Cross-entropy over the selected positions only.
- **Optimization.** AdamW (β₁ 0.9, β₂ 0.999, ε 1e-8), weight decay 0.01, peak LR 5e-5 with
  2,000-step linear warmup then cosine decay to 1e-5, gradient clipping 1.0, bfloat16, dropout
  0.1, seed 42. Effective batch 2,688 sequences — 30,134 optimizer steps per epoch.
- **Hardware.** 4× NVIDIA H200 for pretraining; 4× L40S for all downstream fine-tuning.
- **Architecture.** Mamba layers use a 16-dimensional state space, causal convolution width 4 and
  expansion factor 4, in a pre-normalized residual formulation. Transformer layers use
  padding-aware multi-head self-attention with absolute position embeddings.

## Citation

```bibtex
@article{wang2026limber,
  title   = {LIMBER: A Hybrid RNA Foundation Model for Predicting and Understanding
             RNA Structure and Function},
  author  = {Wang, Ziyao and Qiu, Yunrui and He, Yexiao and Thompson, Isabel and
             Wang, Hongli and Porter, Adam and Li, Ang},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {}
}
```
