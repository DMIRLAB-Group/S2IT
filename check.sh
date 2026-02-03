#!/bin/bash

while true; do
  # 检查 GPU 0, 1, 2 的使用情况
  for gpu_id in 0 1 2; do
    # 使用 nvidia-smi 检查 GPU 的利用率
    utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    
    # 如果利用率为 0，认为该 GPU 空闲
    if [ "$utilization" -eq 0 ]; then
      echo "GPU $gpu_id is idle. Running run.sh with CUDA_VISIBLE_DEVICES=$gpu_id..."
      
      # 设置 CUDA_VISIBLE_DEVICES 环境变量并运行 run.sh
      CUDA_VISIBLE_DEVICES=$gpu_id
      bash run.sh
      
      # 运行完后退出脚本
      exit 0
    fi
  done

  # 如果没有找到空闲的 GPU，等待 10 秒
  echo "No idle GPU found. Retrying in 10 seconds..."
  sleep 10
done