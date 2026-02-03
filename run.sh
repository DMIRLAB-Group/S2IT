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