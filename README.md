# CIFAR-10 图像分类项目

基于深度学习的CIFAR-10图像分类系统，实现了多种CNN架构，包括自定义CNN、ResNet、VGG等模型，并提供完整的训练、评估和可视化功能。

## 项目结构

```
Project3/
├── config.py           # 配置文件（超参数、路径等）
├── data_loader.py      # 数据加载和预处理模块
├── models.py           # 模型定义（多种CNN架构）
├── train.py           # 训练模块
├── evaluate.py        # 评估模块
├── visualize.py       # 可视化模块
├── main.py            # 主程序入口
├── requirements.txt   # Python依赖包
├── README.md          # 本文件
├── data/              # 数据集目录（自动下载）
├── checkpoints/       # 模型检查点保存目录
├── logs/              # 训练日志目录
└── results/           # 结果和可视化保存目录
```

## 功能特性

### 基础功能
- ✅ 多种CNN模型架构（Custom CNN、ResNet18/34、VGG16、MobileNetV2）
- ✅ 自动数据下载和预处理
- ✅ 训练集/验证集自动划分
- ✅ 完整的训练循环和验证流程
- ✅ 测试集评估和性能分析
- ✅ 训练过程可视化（Loss/Accuracy曲线）
- ✅ 混淆矩阵生成和可视化
- ✅ 误分类样本分析

### 高级功能
- ✅ 数据增强（随机裁剪、翻转、Cutout）
- ✅ 正则化技术（Dropout、BatchNorm、Weight Decay）
- ✅ 学习率调度（Cosine Annealing、StepLR、MultiStepLR）
- ✅ 早停机制（Early Stopping）
- ✅ 模型检查点保存和恢复
- ✅ Mixup数据增强（可选）
- ✅ 标签平滑（Label Smoothing，可选）
- ✅ Top-K准确率计算
- ✅ 每类准确率详细分析
- ✅ 混淆类别对分析

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于GPU加速）

## 安装步骤

### 1. 克隆或下载项目

```bash
cd Project3
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 使用方法

### 快速开始（推荐）

运行完整流程（训练 + 评估 + 可视化）：

```bash
python main.py --mode full --model resnet18 --epochs 100
```

### 1. 训练模型

#### 使用默认配置训练

```bash
python main.py --mode train
```

#### 自定义配置训练

```bash
# 使用ResNet18，训练100轮
python main.py --mode train --model resnet18 --epochs 100

# 使用VGG16，自定义学习率和批次大小
python main.py --mode train --model vgg16 --epochs 50 --lr 0.01 --batch-size 64

# 使用自定义CNN
python main.py --mode train --model custom_cnn --epochs 80
```

#### 从检查点恢复训练

```bash
python main.py --mode train --resume
```

### 2. 评估模型

#### 评估最佳模型

```bash
python main.py --mode eval
```

#### 评估指定检查点

```bash
python main.py --mode eval --checkpoint checkpoints/best_model.pth
```

#### 评估并生成可视化

```bash
python main.py --mode eval --visualize
```

### 3. 生成可视化

仅生成可视化图表（使用已保存的最佳模型）：

```bash
python main.py --mode visualize
```

### 4. 支持的模型

- `custom_cnn`: 自定义CNN架构
- `resnet18`: ResNet-18（适配CIFAR-10）
- `resnet34`: ResNet-34（适配CIFAR-10）
- `vgg16`: VGG-16（精简版）
- `mobilenetv2`: MobileNetV2

### 5. 配置参数

主要配置在 `config.py` 中，可以直接修改：

```python
# 训练参数
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 学习率调度
LR_SCHEDULER = 'cosine'  # 'cosine', 'step', 'multistep'

# 早停
EARLY_STOPPING = True
PATIENCE = 15

# 数据增强
DATA_AUGMENTATION = True
CUTOUT = True

# 模型选择
MODEL_NAME = 'resnet18'
```

## 输出结果

### 训练过程

训练过程中会自动保存：

1. **检查点文件**（`checkpoints/`）
   - `best_model.pth`: 验证集上表现最好的模型
   - `last_checkpoint.pth`: 最新的训练检查点

2. **训练日志**（`logs/`）
   - `training_history.json`: 完整的训练历史（loss、accuracy、lr等）

### 评估结果

评估完成后会生成：

1. **评估指标**（`results/`）
   - `evaluation_results.json`: 详细的评估结果
   - `classification_report.txt`: 分类报告

2. **可视化图表**（`results/`）
   - `training_curves.png`: 训练曲线
   - `confusion_matrix.png`: 混淆矩阵
   - `confusion_matrix_normalized.png`: 归一化混淆矩阵
   - `per_class_accuracy.png`: 每类准确率柱状图
   - `misclassified_samples.png`: 误分类样本展示
   - `top_confusion_pairs.png`: 最易混淆的类别对

3. **分析报告**（`results/`）
   - `analysis.md`: 详细的误分类分析报告

## 示例输出

### 训练输出示例

```
==================================================
Configuration:
==================================================
MODEL_NAME                    : resnet18
NUM_EPOCHS                    : 100
BATCH_SIZE                    : 128
LEARNING_RATE                 : 0.1
DEVICE                        : cuda
==================================================

Epoch 1/100 [Train]: 100%|██████| 352/352 [01:23<00:00, loss=1.8234, acc=31.45%]
Epoch 1/100 [Val]:   100%|██████| 40/40 [00:08<00:00, loss=1.5421, acc=42.18%]

Epoch 1/100 Summary:
  Train Loss: 1.8234, Train Acc: 31.45%
  Val Loss:   1.5421, Val Acc:   42.18%
  Learning Rate: 0.100000
Saved best model with validation accuracy: 42.18%
--------------------------------------------------
...
```

### 评估输出示例

```
Test Results:
  Loss: 0.4523
  Accuracy: 91.23%
  Correct: 9123/10000
  Misclassified: 877

Per-class Accuracy:
----------------------------------------
airplane       : 93.20%
automobile     : 95.80%
bird           : 86.70%
cat            : 82.40%
deer           : 89.50%
dog            : 88.30%
frog           : 93.10%
horse          : 94.20%
ship           : 95.60%
truck          : 93.40%
----------------------------------------
```

## 性能优化建议

### 提高准确率

1. **增加训练轮数**：`--epochs 150` 或 `--epochs 200`
2. **使用更深的模型**：`--model resnet34`
3. **调整学习率**：尝试 `--lr 0.05` 或使用学习率预热
4. **启用Mixup**：在 `config.py` 中设置 `MIXUP = True`
5. **标签平滑**：设置 `LABEL_SMOOTHING = 0.1`

### 加速训练

1. **增大批次大小**：`--batch-size 256`（需要足够的GPU内存）
2. **使用GPU**：确保安装了CUDA版本的PyTorch
3. **减少数据增强**：在 `config.py` 中禁用某些增强

### 减少过拟合

1. **增加正则化**：调整 `WEIGHT_DECAY`
2. **启用Dropout**：确保 `USE_DROPOUT = True`
3. **使用早停**：确保 `EARLY_STOPPING = True`
4. **数据增强**：启用 `CUTOUT` 和其他增强

## 常见问题

### 1. CUDA out of memory

解决方法：
- 减小批次大小：`--batch-size 64` 或 `--batch-size 32`
- 使用更小的模型：`--model custom_cnn`

### 2. 数据集下载失败

解决方法：
- 检查网络连接
- 手动下载CIFAR-10数据集并放到 `data/` 目录
- 使用代理或镜像源

### 3. 训练太慢

解决方法：
- 确保使用GPU：检查 `torch.cuda.is_available()`
- 减少训练轮数：`--epochs 50`
- 增加批次大小（如果内存允许）

## 项目亮点

1. **模块化设计**：代码结构清晰，易于扩展和维护
2. **配置灵活**：支持命令行参数和配置文件两种方式
3. **功能完整**：涵盖训练、评估、可视化全流程
4. **注释详细**：每个函数都有清晰的文档字符串
5. **可复现性**：固定随机种子，确保结果可复现
6. **鲁棒性强**：包含错误处理和异常捕获
7. **可视化丰富**：提供多种维度的可视化分析

## 实验建议

### 对比实验

```bash
# 实验1: Custom CNN
python main.py --mode full --model custom_cnn --epochs 100

# 实验2: ResNet18
python main.py --mode full --model resnet18 --epochs 100

# 实验3: VGG16
python main.py --mode full --model vgg16 --epochs 100
```

### 消融实验

```bash
# 测试不同的数据增强策略
# 修改 config.py 中的相关参数后运行

# 测试不同的学习率调度策略
# 修改 LR_SCHEDULER 参数
```

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **数据处理**: NumPy, torchvision
- **可视化**: Matplotlib, Seaborn
- **评估指标**: scikit-learn
- **工具**: tqdm（进度条）

## 参考资料

- CIFAR-10 数据集: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch 官方文档: https://pytorch.org/docs/
- ResNet 论文: "Deep Residual Learning for Image Recognition"
- VGG 论文: "Very Deep Convolutional Networks for Large-Scale Image Recognition"

## 作者

Project3 - CIFAR-10 Image Classification

## 许可证

MIT License

---

**注意**: 首次运行时会自动下载CIFAR-10数据集（约170MB），请确保网络连接正常。
