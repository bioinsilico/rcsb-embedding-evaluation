# RCSB Embedding Evaluation

## Overview
This repository contains scripts and utilities for evaluating the quality of protein structure and sequence embeddings.  
The tools compare embedding-based similarity measures against structural classification benchmarks such as SCOPe or CATH and 
against classical alignment methods (Foldseek, TMalign, Dali, TMvec, BioZernike).

Preprint: [Multi-scale structural similarity embedding search across entire proteomes](https://www.biorxiv.org/content/10.1101/2025.02.28.640875v1).


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

---

## Questions & Issues
For any questions or comments, please open an issue on this repository.

---

## License
This software is released under the BSD 3-Clause License. See the full license text below.

### BSD 3-Clause License

Copyright (c) 2024, RCSB Protein Data Bank, UC San Diego

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


