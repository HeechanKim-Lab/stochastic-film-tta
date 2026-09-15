# Environment Configuration, Demultiplexing Remediation, and Batch Ingestion
**Date:** July 22, 2026  
**Project:** 04_MicroTom_P_Rhizosphere_Amplicon_Analysis
**Target Assays:** Bacterial 16S rRNA (V3–V4) & Fungal ITS2  
**Platform:** Apple Silicon (macOS Sonoma / Darwin 23) via `osx-64` Rosetta 2 Translation  

---

## 1. Computational Architecture & Environment

QIIME 2 distributions rely heavily on compiled C/C++, Fortran, and Cython binaries (including `q2-dada2`, `UniFrac`, and `FastTree`) that are historically compiled and optimized for `x86_64` instruction sets. To ensure binary compatibility and prevent runtime architecture mismatch errors on Apple Silicon, execution is isolated within an emulated `osx-64` Conda prefix under Apple's Rosetta 2 translation layer.

### System Configuration
* **Target Distribution:** QIIME 2 Amplicon (`2024.10.1`)
* **Conda Architecture Override:** Forced `CONDA_SUBDIR=osx-64`
* **Python Runtime:** Python 3.10 (within `qiime2-amp` virtual environment)
* **Solver Mode:** Classic solver with plugins disabled (`CONDA_NO_PLUGINS=true`) to bypass memory overhead during complex SAT-dependency resolution.

---

## 2. Theoretical Rationale & Methodology

### A. Casava 1.8 Semantic Format Requirements
QIIME 2 enforces strict semantic typing and format validation. The input format `CasavaOneEightSingleLanePerSampleDirFmt` requires sequencing files to follow an immutable regular expression:

$$\text{SampleID}\_\text{SampleNumber}\_\text{L001}\_\text{R}[1|2]\_001\text{.fastq.gz}$$

* **Namespace Collisions:** The presence of multiple FastQ pairs bearing identical sample identifiers (e.g., re-sequenced samples across different flowcell lanes or library preparations) breaks the one-sample-per-file-pair invariant and causes directory validation to abort.
* **Deterministic Pairing:** Enforcing standard Casava formatting ensures that downstream demultiplexing artifacts (`SampleData[PairedEndSequencesWithQuality]`) correctly pair forward and reverse reads without index file discrepancies.

### B. FastQ Concatenation across Re-Sequencing Runs
To resolve library depth deficiencies, specific samples underwent supplementary sequencing across separate MiSeq runs.

* **Gzip Stream Concat Validity:** FastQ files are compressed using the DEFLATE algorithm inside the gzip container. Per RFC 1952, concatenated gzip archives are treated as a continuous, valid stream by standard zlib-compatible parsers (`gzip`, `pigz`, `zcat`). Concatenating binary blocks directly via Unix `cat`:
  1. Preserves block headers and bitstreams without requiring decompress/recompress cycles.
  2. Preserves Phred+33 score validity and read integrity.
* **Technical Remediation Performed:**
  * **`Bacteria_GJ2` (`B_J3-4`):** Library `S7` (primary run) and `S79` (supplementary re-run) were concatenated across matching R1 and R2 pairs to reconstitute total sequencing depth. Original run files were archived to `raw_runs/`.
  * **`Bacteria_GB3` (`B_M4-1`):** Sample exhibited three runs (`S45`, `S78`, `S96`). Initial run `S45` failed library generation metrics (under-sequenced/flow cell artifact) and was quarantined to `raw_runs/` to prevent noise introduction into DADA2 error models. Re-runs `S78` and `S96` were concatenated into the primary `S96` identifier.

### C. Batch Isolation Strategy
Sequencing data is partitioned into 12 discrete batches (6 Bacterial 16S, 6 Fungal ITS2):

* Illumina MiSeq error rates fluctuate between physical flowcells, cluster densities, and reagent lots.
* DADA2 uses run-specific parametric error models to learn error transition probabilities ($\mathbb{P}(A \to G)$, $\mathbb{P}(C \to T)$). 
* Ingesting raw FastQ directories into isolated batch artifacts (`demux_16S_{batch}.qza`, `demux_ITS_{batch}.qza`) guarantees that DADA2 error matrices are trained **within-run** rather than artificially conflated across all batches.

---

## 3. Directory Scaffolding & Manifest

The workspace is organized into sequential computational stages:

```text
.
├── 01_imported/     # Raw demultiplexed QIIME 2 artifacts (.qza)
├── 02_trimmed/      # Primer- and heterogeneity spacer-stripped reads
├── 03_denoised/     # DADA2 feature tables and ASV representative sequences
├── 04_merged/       # Consolidated cross-batch tables and sequences
├── 05_taxonomy/     # Taxonomic annotations (SILVA 138 & UNITE v10)
├── 06_phylo/        # Multiple sequence alignments and phylogenetic trees
├── 07_diversity/    # Alpha and beta diversity distance matrices
└── 08_exported/     # Matrices exported for downstream R modeling
```

---

## 4. Ingestion Inventory

All 12 input directories were imported into `SampleData[PairedEndSequencesWithQuality]` semantic containers without format conversion errors.

### Batch Import Registry

| Target Assay | Batch ID | Input Source Directory | Output QIIME 2 Artifact | Import Status |
| :--- | :--- | :--- | :--- | :--- |
| **16S rRNA** | `GB1` | `Bacteria_GB1/` | `01_imported/demux_16S_GB1.qza` | Validated |
| **16S rRNA** | `GB2` | `Bacteria_GB2/` | `01_imported/demux_16S_GB2.qza` | Validated |
| **16S rRNA** | `GB3` | `Bacteria_GB3/` | `01_imported/demux_16S_GB3.qza` | Validated (Remediated) |
| **16S rRNA** | `GJ1` | `Bacteria_GJ1/` | `01_imported/demux_16S_GJ1.qza` | Validated |
| **16S rRNA** | `GJ2` | `Bacteria_GJ2/` | `01_imported/demux_16S_GJ2.qza` | Validated (Remediated) |
| **16S rRNA** | `GJ3` | `Bacteria_GJ3/` | `01_imported/demux_16S_GJ3.qza` | Validated |
| **ITS2** | `GB1` | `Fungi_GB1/` | `01_imported/demux_ITS_GB1.qza` | Validated |
| **ITS2** | `GB2` | `Fungi_GB2/` | `01_imported/demux_ITS_GB2.qza` | Validated |
| **ITS2** | `GB3` | `Fungi_GB3/` | `01_imported/demux_ITS_GB3.qza` | Validated |
| **ITS2** | `GJ1` | `Fungi_GJ1/` | `01_imported/demux_ITS_GJ1.qza` | Validated |
| **ITS2** | `GJ2` | `Fungi_GJ2/` | `01_imported/demux_ITS_GJ2.qza` | Validated |
| **ITS2** | `GJ3` | `Fungi_GJ3/` | `01_imported/demux_ITS_GJ3.qza` | Validated |
