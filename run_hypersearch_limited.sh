#!/bin/bash
# 限制配置数量的超参数搜索脚本

echo "开始 Wide ResNet Small 有限超参数搜索..."
echo "========================================"
echo "将从所有配置中随机选择最多20个进行测试"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sole

# 限制模式（100个epoch，最多20个配置）
python hyperparameter_search.py \
    --gpu-id 0 \
    --epochs 100 \
    --max-configs 20

echo ""
echo "有限搜索完成!"
