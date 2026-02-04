# Getting Started

S2IT runs dependency parsing, builds
train/dev data, trains the model, performs extraction and classification, then
evaluates results.

## Prerequisites

- Python environment:
  - Install dependencies: `pip install -r requirements.txt`
- Stanza model files:
  - `ACOS-main/preprocess_dependency.py` uses `stanza`
  - Ensure English models are available at `stanza_resources/stanza-en/`

## One-Click Run

Run `run.sh` (default `dataset=restaurant`):

```bash
bash run.sh
```

The script does the following:

```bash
export CUDA_VISIBLE_DEVICES=

export model_name_or_path=Qwen/Qwen2.5-7B-Instruct
export template=qwen
python ACOS-main/preprocess_dependency.py
python ACOS-main/preprocess_training.py
python ACOS-main/preprocess_dev.py

export dataset=restaurant
bash train.sh
bash extraction.sh
python ACOS-main/pre_classification.py
bash classification.sh
python evaluate.py
```
