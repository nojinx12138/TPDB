# TPDB benchmark

Classification and regression benchmarks using similarity-controlled splits, 26 models and seeds 0–4.

## Data

Download data from **https://tpdb.cc/downloads/**.

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

These commands preview the run. Add `--execute` to start training. Use `--group tabular`, `deep_strings`, `pretrained_lm` or `graphs` to select a model group; `--gpu` selects the GPU.

Prepare Half-life SMILES inputs with `regression/half_life_smiles/code/prepare_half_life_smiles.py`, then use the `half-life-smiles` command. Run the preparation script with `--help` for its arguments. Use `tools/stage_runtime.py` to prepare a runtime project.

## Results

`results/` contains five-seed summaries, rankings, MCC matrices and length-group AUC summaries.

Result tables use **TR**, **PU**, **BI**, **MS**, with train–test combinations such as `MS_BI`. To export new run results with these names:

```bash
python tools/format_results.py /path/to/run/results --output-dir /path/to/manuscript_results
```

`backbones.csv` lists pretrained model IDs and input settings. Download weights from `https://huggingface.co/{model_id}`. For ProtBERT, cache `config.json`, `vocab.txt` and `pytorch_model.bin` together.
