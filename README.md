# 启动说明

这份说明对应 `run.sh` 的流程，按顺序完成依存解析、训练集/验证集构建、训练、抽取、分类与评测。

## 运行前准备

- Python 环境与依赖：
  - 安装依赖：`pip install -r requirements.txt`
- 依存解析模型（Stanza）：
  - `ACOS-main/preprocess_dependency.py` 使用 `stanza`，需要英文模型文件位于 `stanza_resources/stanza-en/`

## 一键运行

直接运行 `run.sh`（默认 `dataset=restaurant`）：

```bash
bash run.sh
```

`run.sh` 的默认流程如下：

```bash
export CUDA_VISIBLE_DEVICES=0

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
