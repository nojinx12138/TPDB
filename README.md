# TPDB benchmark

Classification and regression benchmarks using similarity-controlled splits, 26 models and seeds 0–4.

## Data

Download data from **https://tpdb.cc/downloads/**. Data, trained weights and full per-peptide predictions are not included in this repository.

Classification uses four negative-sampling schemes:

- **TR**: traditional negative sampling.
- **PU**: positive–unlabeled learning-based negative sampling.
- **BI**: biologically constrained negative sampling.
- **MS**: mixed negatives from TR, PU and BI.

Data folders `A`, `B`, `C`, `D` correspond to **TR**, **PU**, **BI**, **MS**, respectively. Each uses `ACTIVITY/{A,B,C,D}/train.csv` and `validation.csv`; TR/PU/BI use their corresponding `test.csv`. MS is evaluated on the three TR/PU/BI test sets.

Regression uses `{half_life_len50,caco2_len50,pampa_len50}/{train,valid,test}.csv` with `input,y` columns.

## Run

Use Linux x86_64, Python 3.10.20 and PyTorch 2.11.0 with CUDA 12.8. Create the environment with:

```bash
conda env create -f environment.yml
conda activate tpdb-benchmark
```

`requirements.txt` lists the primary dependencies. `environment.yml` also pins their runtime dependencies and includes the PyTorch CUDA wheel index.

```bash
python tools/run_experiment.py classification --dataset-root /path/to/homology_30_split --output-root /path/to/new_classification_run
python tools/run_experiment.py regression --dataset-root /path/to/regression_similarity_controlled --output-root /path/to/new_regression_run
```

These commands print the exact training commands. Add `--execute` to run. Use `--group tabular`, `deep_strings`, `pretrained_lm` or `graphs` for separate workers; `--gpu` selects the GPU. Outputs should use a new directory. Checkpoints retain the original best-validation saving/loading and resume behavior.

For the Half-life SMILES extension, use `half-life-smiles` instead of `regression`, with the derived Half-life SMILES input directory. `regression/half_life_smiles/code/prepare_half_life_smiles.py` preserves the original splits using RDKit FASTA-to-canonical-SMILES conversion; its `--help` lists the required finalized baseline project inputs. `tools/stage_runtime.py` stages code into a fresh runtime project when needed.

## Results

`results/` contains five-seed summaries, rankings, MCC matrices and length-group AUC summaries. **GIN-atom** uses the edge-aware **GINE** implementation.

Result tables use **TR**, **PU**, **BI**, **MS**, with train–test combinations such as `MS_BI`. To export new run results with these names:

```bash
python tools/format_results.py /path/to/run/results --output-dir /path/to/manuscript_results
```

`backbones.csv` lists the six frozen backbones, Hugging Face model/tokenizer IDs, available revisions, embedding dimensions and effective token limits. Obtain weights from `https://huggingface.co/{model_id}`; keep ProtBERT's `config.json`, `vocab.txt` and `pytorch_model.bin` together in its Hugging Face cache snapshot. PepDoRA is a DoRA adapter on `DeepChem/ChemBERTa-77M-MLM` (r=64, alpha=128, dropout=0.1, targets=query/key/value). All backbones are frozen and use attention-mask mean pooling. Only the downstream MLP is trained; its settings are in the training code.
