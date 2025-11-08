# 超参数优化搜索指南

本项目提供了自动化的超参数优化功能，可以系统地测试不同配置下 Wide ResNet Small 模型的性能。

## 功能特性

### 1. GPU选择功能

现在可以指定使用哪个GPU进行训练：

```bash
# 使用GPU 0
python main.py --mode train --gpu-id 0

# 使用GPU 1
python main.py --mode train --gpu-id 1
```

在 `config.py` 中也可以设置默认GPU：
```python
GPU_ID = 0  # 默认使用的GPU编号
```

### 2. 超参数搜索空间

搜索脚本会自动测试以下超参数组合：

#### Batch Size
- 64
- 128
- 256

#### 正则化系数 (Weight Decay)
- 1e-4
- 5e-4
- 1e-3

#### 数据增强
- Cutout (length=12) + Random Flip
- Cutout (length=16) + Random Flip
- No Cutout + Random Flip

#### 训练技巧
- 无特殊技巧
- Mixup (alpha=1.0)
- Label Smoothing (epsilon=0.1)

#### 学习率调度
- Cosine Annealing (lr=0.1)
- MultiStep (lr=0.1, milestones=[60,120,160])

**总共配置组合数**: 3 × 3 × 3 × 3 × 2 = **162种配置**

## 使用方法

### 快速测试模式

适用于快速验证脚本是否正常工作：

```bash
# 赋予执行权限
chmod +x run_hypersearch_quick.sh

# 运行快速测试
./run_hypersearch_quick.sh
```

快速模式特点：
- 只训练 10 个 epoch
- 只测试 3 个随机配置
- 用时约 30-60 分钟

### 有限搜索模式（推荐）

从所有配置中随机选择一部分进行测试：

```bash
# 赋予执行权限
chmod +x run_hypersearch_limited.sh

# 运行有限搜索（测试20个配置）
./run_hypersearch_limited.sh
```

有限模式特点：
- 训练 100 个 epoch
- 测试 20 个随机配置
- 用时约 10-15 小时

### 完整搜索模式

测试所有配置组合：

```bash
# 赋予执行权限
chmod +x run_hypersearch_full.sh

# 运行完整搜索
./run_hypersearch_full.sh
```

⚠️ **警告**：完整模式特点：
- 训练 100 个 epoch
- 测试所有 162 个配置
- 预计用时 **80+ 小时**（约3-4天）

### 自定义搜索

使用 Python 脚本直接运行，可以自定义参数：

```bash
# 激活环境
conda activate sole

# 自定义参数
python hyperparameter_search.py \
    --gpu-id 0 \           # 使用的GPU编号
    --epochs 50 \          # 训练轮数
    --max-configs 10       # 最多测试的配置数
```

## 结果输出

### 目录结构

所有实验结果会保存在时间戳命名的目录中：

```
results/
└── hypersearch_1108_1430/          # 搜索总目录
    ├── exp_001/                     # 实验1
    │   ├── checkpoints/             # 模型检查点
    │   │   └── best_model.pth
    │   ├── logs/                    # 训练日志
    │   │   └── training_history.json
    │   ├── hyperparameters.json     # 该实验的超参数
    │   ├── confusion_matrix.png     # 混淆矩阵
    │   ├── classification_report.txt
    │   ├── evaluation_results.json
    │   └── analysis.md              # 分析报告
    ├── exp_002/
    │   └── ...
    ├── exp_003/
    │   └── ...
    └── search_summary.json          # 所有实验的汇总
```

### 查看结果

1. **搜索摘要**
   
   查看 `search_summary.json` 获取所有实验的概览：
   ```bash
   cat results/hypersearch_*/search_summary.json
   ```

2. **最佳配置**
   
   脚本运行结束时会自动打印 Top 5 最佳配置

3. **详细分析**
   
   每个实验目录中的 `analysis.md` 包含详细的分析报告

## 脚本说明

### hyperparameter_search.py

主要的超参数搜索脚本，功能包括：

- ✅ 自动生成所有配置组合
- ✅ 为每个配置创建独立的目录
- ✅ 自动训练、评估和可视化
- ✅ 保存所有超参数和结果
- ✅ 生成搜索摘要报告
- ✅ 支持GPU选择
- ✅ 错误处理和恢复

### 命令行参数

```
--gpu-id INT          使用的GPU编号（默认：0）
--max-configs INT     最多测试的配置数量（默认：全部）
--epochs INT          每个实验的训练轮数（默认：100）
--quick-test          快速测试模式
```

## 监控进度

### 实时监控

可以使用多种方式监控训练进度：

```bash
# 查看当前正在运行的实验
ps aux | grep python

# 监控GPU使用情况
watch -n 1 nvidia-smi

# 查看最新的日志
tail -f results/hypersearch_*/exp_*/logs/training_history.json
```

### 中断和恢复

如果需要中断搜索：
- 按 `Ctrl+C` 停止
- 已完成的实验结果会被保存
- 可以修改脚本从特定实验继续

## 结果分析

### Python 脚本分析

创建一个分析脚本来比较所有实验：

```python
import json
import pandas as pd

# 读取搜索摘要
with open('results/hypersearch_1108_1430/search_summary.json') as f:
    summary = json.load(f)

# 转换为DataFrame
experiments = []
for exp in summary['experiments']:
    if 'error' not in exp:
        experiments.append({
            'name': exp['experiment_name'],
            'test_acc': exp['test_acc'],
            'val_acc': exp['best_val_acc'],
            'batch_size': exp['config']['batch_size'],
            'weight_decay': exp['config']['weight_decay'],
            # ... 其他字段
        })

df = pd.DataFrame(experiments)
df = df.sort_values('test_acc', ascending=False)
print(df.head(10))
```

## 注意事项

1. **存储空间**: 每个实验约需要 100-500 MB，确保有足够的磁盘空间
2. **GPU内存**: 确保GPU有足够内存运行实验（建议至少6GB）
3. **时间估算**: 每个配置约需要 30-60 分钟完成100个epoch的训练
4. **备份结果**: 定期备份 `results` 目录中的重要实验结果

## 常见问题

### Q: 如何使用多个GPU？

修改脚本在不同GPU上并行运行：

```bash
# 终端1 - GPU 0
python hyperparameter_search.py --gpu-id 0 --max-configs 10 &

# 终端2 - GPU 1
python hyperparameter_search.py --gpu-id 1 --max-configs 10 &
```

### Q: 如何修改搜索空间？

编辑 `hyperparameter_search.py` 中的 `define_search_space()` 方法，添加或修改参数值。

### Q: 如何只测试特定的配置？

可以修改 `generate_configs()` 方法，或直接在脚本中调用 `run_experiment()` 测试单个配置。

## 示例输出

```
================================================================================
超参数搜索完成!
================================================================================

成功完成的实验: 20/20

Top 5 最佳配置:
--------------------------------------------------------------------------------

第 1 名:
  实验名称: exp_015
  测试准确率: 95.67%
  验证准确率: 95.45%
  配置:
    - Batch Size: 128
    - Weight Decay: 0.0005
    - Cutout: True (length=16)
    - Mixup: False
    - Label Smoothing: 0.1
    - LR Scheduler: cosine
  结果目录: results/hypersearch_1108_1430/exp_015
...
```

## 技术支持

如有问题，请检查：
1. conda环境是否正确激活
2. GPU驱动和CUDA是否正常
3. 磁盘空间是否充足
4. 查看具体实验目录中的日志文件
