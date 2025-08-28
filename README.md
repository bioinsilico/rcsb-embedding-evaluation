# RCSB Embedding Evaluation

This repository contains scripts and utilities for evaluating the quality of protein structure and sequence embeddings.  
The tools compare embedding-based similarity measures against structural classification benchmarks such as SCOPe or CATH and 
against classical alignment methods (Foldseek, TMalign, Dali, TMvec, BioZernike).

## Features
- Precision–recall benchmarking of embedding scores (`src/pr_bench.py` and related scripts).
- Query sensitivity analysis across SCOPe or CATH levels (`src/qs_bench.py` and variants).
- Dataset loaders for domain labels and score files under `src/analysis`.
- Utilities for computing cosine similarity scores from embedding files (`src/scripts/compute_embedding_scores.py`).
- Helper scripts for processing AlphaFold outputs and alignment tool results (`src/scripts`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/rcsb-embedding-evaluation.git
   cd rcsb-embedding-evaluation
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib scikit-learn scipy tqdm
   ```
   Adjust the list of packages as needed for your environment.

## Data Preparation
- **Embeddings**: Each domain or chain embedding is stored as a CSV file containing the vector for that entity.
- **Domain classes**: A tab-separated file mapping domain IDs to SCOPe or CATH classes (e.g., `d1a0b__\t1.10.8.10`).
- **Score files**: Pairwise similarity scores between domains. These can be generated from embeddings or produced by alignment tools.

Compute embedding-based scores:
```bash
python src/scripts/compute_embedding_scores.py \
  --embeddings-path embeddings \
  --score-file structure_scores.tsv
```

## Precision–Recall Benchmark
Generate precision–recall curves at different SCOPe depths:
```bash
python src/pr_bench.py \
  --structure-embeddings-score structure_scores.tsv \
  --sequence-embeddings-score sequence_scores.tsv \
  --mean-embeddings-score mean_scores.tsv \
  --domain-class-file scop_map.tsv \
  --results-path benchmarks \
  --out-path plots
```
The `results-path` directory should contain baseline score files (e.g., `foldseek.txt`, `TMalign.txt`, `dali.txt`, `tmvec.txt`, `pdb-foldseek-zer.txt`).
Output plots are written to `out-path`.

Other scripts such as `pr_chain_bench.py`, `pr_assembly_bench.py`, and `pr_af_bench.py` provide analogous evaluations for different datasets.

## Query Sensitivity Benchmark
Plot query sensitivity curves:
```bash
python src/qs_bench.py \
  --structure-embeddings-score structure_scores.tsv \
  --sequence-embeddings-score sequence_scores.tsv \
  --mean-embeddings-score mean_scores.tsv \
  --results-path benchmarks \
  --domain-classes scop_map.tsv \
  --out-path plots
```

## Additional Tools
- `src/fp-analysis.py` – plot false-positive score distributions.
- `src/scripts/get_af_embeddings.py` and `src/scripts/merge_af_cath.py` – work with AlphaFold embeddings and CATH labels.
- `src/scripts/process_foldseek_multimer_results.py` and related parsers – convert outputs from alignment tools into the required TSV format.

## License
No license file is present.  Contact the repository authors for licensing information.

