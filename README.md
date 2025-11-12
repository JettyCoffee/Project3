# Project 3 - CIFAR-10 图像分类项目

[![Python](https://img.shields.io/badge/python-3.13.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.9.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 基于现代卷积神经网络架构和先进训练技术的高性能 CIFAR-10 图像分类深度学习项目

本项目在 CIFAR-10 数据集上实现了一个综合性的图像分类系统，使用 Wide ResNet 配合先进的正则化和优化技术，达到了 **96.45% 的测试准确率**。项目包含多个最先进的模型架构、完善的训练工具以及全面的评估和可视化功能。

## 目录

- [使用方法](#使用方法)
  - [快速开始](#快速开始)
  - [模型训练](#模型训练)
  - [模型评估](#模型评估)
  - [结果可视化](#结果可视化)
  - [完整流程](#完整流程)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [开源协议](#开源协议)

## 使用方法

安装依赖包：
```bash
pip install -r requirements.txt
```

### 快速开始

使用默认设置训练 Wide ResNet 模型并在测试集上评估：

```bash
python main.py --mode full --model wide_resnet --epochs 100 --batch-size 256
```

该命令将执行以下操作：
1. 自动下载 CIFAR-10 数据集
2. 训练模型 100 个周期
3. 在测试集上进行评估
4. 生成可视化结果和分析报告

### 模型训练

使用自定义超参数训练模型：

```bash
# 使用自定义学习率训练 ResNet-50
python main.py --mode train --model resnet50 --epochs 100 --batch-size 256 --lr 0.1

# 从检查点恢复训练
python main.py --mode train --resume

# 在指定 GPU 上训练
python main.py --mode train --gpu-id 0
```

可用的模型：`resnet18`、`resnet34`、`resnet50`、`wide_resnet`、`dla34`、`vit`

### 模型评估

在测试集上评估已训练的模型：

```bash
# 使用检查点路径进行评估
python main.py --mode eval --checkpoint checkpoints/best_model_{model}_{timestamp}.pth
```

### 结果可视化

为已训练的模型生成可视化结果：

```bash
python main.py --mode visualize --checkpoint checkpoints/best_model_{model}_{timestamp}.pth
```

生成的可视化包括：
- 训练曲线（损失和准确率）
- 混淆矩阵
- 各类别准确率柱状图
- 误分类样本分析
- Grad-CAM 热力图

### 完整流程

运行完整的训练、评估和可视化流程：

```bash
python main.py --mode full --model wide_resnet --epochs 100 --batch-size 256
```

## 项目结构

```
Project3/
├── main.py                 # 主程序入口，包含命令行参数解析
├── config.py              # 配置文件和超参数设置
├── models.py              # 模型架构（ResNet、WRN、DLA、ViT）
├── data_loader.py         # 数据加载和数据增强
├── train.py               # 训练循环和早停机制
├── evaluate.py            # 模型评估和性能指标
├── visualize.py           # 可视化工具
├── grad_cam.py            # Grad-CAM 实现
├── requirements.txt       # Python 依赖包列表
├── README.md             # 本文件
```

## 模型架构

### 已实现的模型架构

| 模型 | 参数量 | 测试准确率 | 训练时间 |
|-------|-----------|---------------|---------------|
| Wide ResNet-28-10 | ~3650万 | **96.45%** | ~8小时（100轮）|
| ResNet-50 | ~2350万 | 94.23% | ~6小时（100轮）|
| DLA-34 | ~1570万 | 93.52% | ~5小时（100轮）|
| ResNet-34 | ~2130万 | 93.18% | ~5小时（100轮）|
| ResNet-18 | ~1120万 | 92.74% | ~4小时（100轮）|
| ViT-Tiny | ~570万 | 89.31% | ~10小时（100轮）|

## 配置说明

关键超参数可以在 `config.py` 中修改：

```python
# 训练超参数
BATCH_SIZE = 256           # 训练批次大小
NUM_EPOCHS = 100          # 训练轮数
LEARNING_RATE = 0.1       # 初始学习率
MOMENTUM = 0.9            # SGD 动量因子
WEIGHT_DECAY = 0.0005     # L2 正则化系数

# 数据增强
RANDOM_CROP = True        # 随机裁剪（带填充）
RANDOM_HORIZONTAL_FLIP = True  # 随机水平翻转
CUTOUT = True             # Cutout 正则化
CUTOUT_LENGTH = 16        # Cutout 区域大小

# 正则化
DROPOUT_RATE = 0.5        # Dropout 概率
LABEL_SMOOTHING = 0.1     # 标签平滑因子

# 早停机制
EARLY_STOPPING = True     # 启用早停
PATIENCE = 25             # 早停容忍轮数
MIN_DELTA = 0.001         # 最小改善阈值
```

## 实验结果

### 最佳模型性能（Wide ResNet-28-10）

- **测试准确率**：96.45%
- **Top-3 准确率**：99.55%
- **Top-5 准确率**：99.75%

### 各类别准确率

| 类别 | 准确率 |
|-------|----------|
| 青蛙（Frog）| 98.5% |
| 船（Ship）| 98.1% |
| 汽车（Automobile）| 97.9% |
| 马（Horse）| 97.9% |
| 卡车（Truck）| 97.5% |
| 鹿（Deer）| 97.2% |
| 飞机（Airplane）| 96.7% |
| 鸟（Bird）| 95.3% |
| 猫（Cat）| 93.7% |
| 狗（Dog）| 91.7% |

### 常见误分类情况

1. **狗 ↔ 猫**（6.3% 和 3.4%）：相似的毛发纹理和姿态
2. **飞机 ↔ 船**（1.6%）：在低分辨率下具有相似的几何形状
3. **卡车 ↔ 汽车**（1.6%）：相似的车辆结构

所有评估结果，包括混淆矩阵、训练曲线和 Grad-CAM 可视化，都会保存在 `results/` 目录中。

## 开源协议

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。

---

**作者**：陈子谦 
**日期**：2025年11月13日
**课程**：当代人工智能 - Project3
