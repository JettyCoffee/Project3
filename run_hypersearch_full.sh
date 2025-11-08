#!/bin/bash
# 完整超参数搜索脚本

echo "开始 Wide ResNet Small 完整超参数搜索..."
echo "========================================"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sole

# 完整模式（100个epoch，所有配置组合）
# 注意：这将测试 3x3x3x3x2 = 162 个配置组合
# 如果每个实验需要30分钟，总共需要约81小时
python hyperparameter_search.py \
    --gpu-id 0 \
    --epochs 100

echo ""
echo "完整搜索完成!"
