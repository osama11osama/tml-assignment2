# Trustworthy Machine Learning — Stolen / Derived Model Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Research](https://img.shields.io/badge/topic-model%20provenance-purple)](#project-overview)
[![Use](https://img.shields.io/badge/use-research%20%26%20education-orange)](#responsible-use--legal-notice)

A research project for studying **model provenance and stolen/derived model detection**.

The core question is:

> **Given a reference model and a collection of suspect models, can we identify which suspects were copied from, fine-tuned from, or otherwise derived from the reference?**

This repository was developed in an academic Trustworthy Machine Learning setting and is published as a **research, learning, and portfolio artifact**.

> Use this project only on models and data that you own or are explicitly authorized to analyze.

---

## Project overview

The project evaluates multiple signals for detecting model derivation, including:

- parameter-space similarity;
- logit/output similarity;
- probe-based behavioral comparison;
- multi-stage forensic features;
- ensemble scoring;
- reproducible ranking and evaluation pipelines.

The original experiment used ResNet-18 models and a CIFAR-100-related setup with hundreds of suspect models.

The goal is not simply to classify models, but to explore what traces of lineage can remain after copying, retraining, or modifying a neural network.

---

## Why this matters

Model provenance is relevant to:

- intellectual-property protection;
- model theft detection;
- supply-chain and artifact provenance;
- ML forensics;
- model governance;
- watermarking/fingerprinting research;
- understanding how transformations affect detectable similarity.

A high similarity score alone is not proof of unlawful copying. Legitimate models can share architectures, datasets, initialization strategies, checkpoints, or training pipelines. Results therefore need to be interpreted in context and alongside other evidence.

---

## Research approaches

### 1. Weight-space similarity

A simple baseline compares model parameters directly, for example through cosine similarity or related statistics.

This can be useful when a suspect is a close copy, but may become less reliable after substantial fine-tuning, pruning, reinitialization, or architectural changes.

### 2. Behavioral / logit similarity

Instead of comparing raw weights, the project evaluates how different models respond to the same probe inputs.

This helps capture functional similarity even when parameters have changed.

### 3. Multi-stage forensic ensemble

The more advanced pipeline combines several forms of evidence into a single score for each suspect model.

Conceptually:

```text
Reference model
      ↓
Generate reference features on probe inputs
      ↓
Evaluate each suspect model
      ↓
Extract multiple similarity / behavior features
      ↓
Normalize and combine evidence
      ↓
Rank suspects by likelihood of derivation
```

Implementation details are documented in:

- [`docs/METHOD.md`](docs/METHOD.md)
- [`docs/FORENSIC_STRATEGY.md`](docs/FORENSIC_STRATEGY.md)
- [`experiments/exp_notes.md`](experiments/exp_notes.md)

---

## Repository structure

```text
tml-assignment2/
├── README.md
├── requirements.txt
├── src/                    # shared model/data utilities
├── scripts/                # scoring and forensic pipelines
├── configs/                # experiment configuration
├── experiments/            # experiment notes and snapshots
├── results/                # generated/local result structure
├── docs/                   # method and infrastructure notes
├── report/                 # academic report source
└── data/                   # metadata used by the experiment
```

Large model weights are intended to remain outside Git and should be obtained only from lawful, permitted sources.

---

## Methods implemented

| Version | Method | Main idea |
|---|---|---|
| `v001` | Weight cosine similarity | Compare model parameters directly |
| `v002` | Logit similarity | Compare model outputs on shared probes |
| `v003` | Forensic ensemble | Combine multiple behavioral/structural signals |

The exact experimental scripts are kept under `scripts/` so that results can be reproduced and extended.

---

## Recorded result

One public leaderboard snapshot recorded in this repository was:

**TPR @ 5% FPR: `0.537037`**

for the logit-similarity experiment tagged:

```text
v0.2-logit-0.537
```

This number belongs to the original course/evaluation environment. It should **not** be interpreted as a general detection rate for arbitrary real-world models.

---

## Getting started

Create an environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then use the project scripts with models and datasets that you are permitted to analyze.

Typical experimental flow:

```text
prepare authorized models/data
        ↓
precompute reference-model features
        ↓
extract suspect-model features
        ↓
combine / score evidence
        ↓
evaluate ranking
```

GPU acceleration is recommended for large suspect sets.

---

## Reproducibility

Reproduction may depend on:

- exact model checkpoints;
- probe samples;
- architecture and preprocessing;
- framework version;
- floating-point differences;
- GPU/software environment;
- random seeds;
- normalization and ensemble parameters.

For that reason, the repository preserves experiment notes and configuration files rather than only the final scores.

Some infrastructure notes reference university or cloud-GPU environments used during experimentation. Those documents are historical/reproducibility material and may require adaptation to another environment.

---

## Defensive and governance perspective

This project can also be viewed from a defensive perspective: model owners may want to understand whether their models remain recognizable after copying or modification.

Related research directions include:

- model watermarking;
- model fingerprinting;
- provenance metadata;
- signed model artifacts;
- secure model distribution;
- access logging and API rate controls;
- contractual and governance controls;
- robust evidence standards for ownership disputes.

Technical similarity analysis should generally be treated as one piece of evidence rather than a complete legal conclusion.

---

## Responsible use & legal notice

This repository is provided for **lawful educational, academic, defensive, provenance, and authorized ML-security research purposes only**.

By using the project, you are responsible for ensuring compliance with applicable laws, contracts, licenses, institutional policies, intellectual-property rules, privacy requirements, and the scope of any authorization you received.

You must not use this project to:

- access or analyze non-public models without authorization;
- violate model, dataset, platform, or API terms;
- bypass technical access controls;
- claim ownership of models or data based solely on this project's output;
- misuse similarity scores as definitive proof of theft;
- interfere with third-party services or infrastructure;
- use publication of this repository as permission to investigate a third party.

The author does **not** authorize unlawful or unauthorized use and is not responsible for how third parties choose to use, modify, or redistribute this work.

### Disclaimer of warranty and liability

All code, research notes, experimental results, and documentation are provided **"as is"**, without warranties or guarantees of any kind.

Scores and conclusions may be incomplete, environment-dependent, statistically uncertain, or unsuitable for a particular operational or legal purpose.

To the maximum extent permitted by applicable law, the author shall not be liable for losses, damages, claims, IP disputes, business consequences, service disruption, data loss, or other consequences arising from use, misuse, modification, or redistribution of this project.

Nothing in this repository constitutes legal advice or a legal determination of model ownership. No disclaimer can guarantee complete exclusion of liability in every jurisdiction.

---

## Academic integrity

This repository originated from academic work and is published for learning, reproducibility, and portfolio purposes.

If you are enrolled in a course with a similar assignment:

- follow your institution's academic-integrity rules;
- do not submit this implementation as your own where reuse is prohibited;
- cite reused code and ideas where appropriate;
- ask the instructor if consulting public solutions is permitted.

A public repository does not override course rules.

---

## Author

**Osama Altamar**  
Cybersecurity and software engineering — interests include ML security, provenance, privacy, secure systems, and defensive research.

GitHub: [@osama11osama](https://github.com/osama11osama)

---

## Related projects

- [`tml-assignment1`](https://github.com/osama11osama/tml-assignment1) — membership inference / ML privacy
- [`tml-assignment3`](https://github.com/osama11osama/tml-assignment3) — adversarial robustness
- [`Sidechannel-timing-attack-starter`](https://github.com/osama11osama/Sidechannel-timing-attack-starter) — timing side-channel demonstration
