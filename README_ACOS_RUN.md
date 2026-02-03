# 启动说明

这份说明对应 `run.sh` 的流程，按顺序完成依存解析、训练集/验证集构建、训练、抽取、分类与评测。

## 运行前准备

- Python 环境与依赖：
  - 建议 Python 3.8+
  - 安装依赖：`pip install -r requirements.txt`
- 依存解析模型（Stanza）：
  - `ACOS-main/preprocess_dependency.py` 使用 `stanza`，需要英文模型文件位于 `stanza_resources/stanza-en/`
  - 如未下载，请先准备好该目录（离线环境可手动拷贝）

## 一键运行

直接运行 `run.sh`（默认 `dataset=restaurant`）：

```bash
bash run.sh
```

`run.sh` 的默认流程如下（等价于逐步执行）：

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

## 常见调整

- 切换数据集为 `laptop`：
  - 方法 1：编辑 `run.sh` 中的 `export dataset=restaurant`
  - 方法 2：不使用 `run.sh`，按上面的步骤手动执行并设置 `export dataset=laptop`
- 切换模型或模板：
  - 修改 `model_name_or_path` 与 `template` 的值即可

## 主要输出位置

- 训练输出：`path_to_sft_checkpoint_${dataset}/`
- 抽取预测：`ao_result/generated_predictions.jsonl`
- 分类预测：`${dataset}_result/generated_predictions.jsonl`
- 评测结果：`python evaluate.py` 输出精确率/召回率/F1
