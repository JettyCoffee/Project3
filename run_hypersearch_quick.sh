#!/bin/bash
# 快速超参数搜索测试脚本

echo "开始 Wide ResNet Small 超参数搜索..."
echo "========================================"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sole

# 快速测试模式（10个epoch，3个配置）
python hyperparameter_search.py \
    --gpu-id 0 \
    --quick-test

echo ""
echo "快速测试完成!"
