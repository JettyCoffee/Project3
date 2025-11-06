"""
可视化模块 - 实现训练曲线、混淆矩阵、误分类样本等可视化
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
import os
from sklearn.metrics import confusion_matrix

from config import Config


def plot_training_curves(history_path=None, save_path=None):
    """
    绘制训练曲线（损失和准确率）
    
    Args:
        history_path: 训练历史JSON文件路径
        save_path: 保存图片的路径
    """
    if history_path is None:
        history_path = os.path.join(Config.LOG_DIR, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"History file {history_path} not found.")
        return
    
    # 加载训练历史
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(list(epochs))  # x轴以1为间隔

    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(list(epochs))  # x轴以1为间隔

    # 学习率曲线
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(list(epochs))  # x轴以1为间隔

    # 训练-验证准确率差异
    acc_diff = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, acc_diff, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference (%)')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(list(epochs))  # x轴以1为间隔
    
    plt.tight_layout()
    
    # 保存
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
        normalize: 是否归一化
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path is None:
        suffix = '_normalized' if normalize else ''
        save_path = os.path.join(Config.RESULTS_DIR, f'confusion_matrix{suffix}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_per_class_accuracy(per_class_acc, save_path=None):
    """
    绘制每个类别的准确率柱状图
    
    Args:
        per_class_acc: 每个类别的准确率数组
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(per_class_acc / per_class_acc.max())
    bars = plt.bar(Config.CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor='black')
    
    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)
    
    # 添加平均线
    mean_acc = per_class_acc.mean() * 100
    plt.axhline(y=mean_acc, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'per_class_accuracy.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to {save_path}")
    
    plt.close()


def plot_misclassified_samples(misclassified_samples, num_samples=20, save_path=None):
    """
    可视化误分类样本
    
    Args:
        misclassified_samples: 误分类样本列表
        num_samples: 要显示的样本数
        save_path: 保存路径
    """
    num_samples = min(num_samples, len(misclassified_samples))
    
    if num_samples == 0:
        print("No misclassified samples to plot.")
        return
    
    # 计算网格大小
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    fig.suptitle('Misclassified Samples', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if idx < num_samples:
            sample = misclassified_samples[idx]
            
            # 反归一化图像
            img = sample['image'].numpy()
            img = img.transpose(1, 2, 0)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            true_label = Config.CLASS_NAMES[sample['true_label']]
            pred_label = Config.CLASS_NAMES[sample['pred_label']]
            confidence = sample['probs'][sample['pred_label']]
            
            title = f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})"
            ax.set_title(title, fontsize=9, color='red')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'misclassified_samples.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Misclassified samples plot saved to {save_path}")
    
    plt.close()


def plot_top_confusion_pairs(confusion_pairs, top_k=10, save_path=None):
    """
    绘制最容易混淆的类别对
    
    Args:
        confusion_pairs: 混淆对列表
        top_k: 显示前k个
        save_path: 保存路径
    """
    top_pairs = confusion_pairs[:top_k]
    
    labels = [f"{p['true_class']}\n→{p['pred_class']}" for p in top_pairs]
    counts = [p['count'] for p in top_pairs]
    percentages = [p['percentage'] for p in top_pairs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Top {top_k} Confusion Pairs', fontsize=16, fontweight='bold')
    
    # 按数量
    colors1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(counts)))
    bars1 = ax1.barh(labels, counts, color=colors1, edgecolor='black')
    ax1.set_xlabel('Count', fontsize=12)
    ax1.set_title('By Absolute Count', fontsize=14)
    ax1.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax1.text(count, bar.get_y() + bar.get_height()/2, 
                f' {count}', va='center', fontsize=10)
    
    # 按百分比
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.8, len(percentages)))
    bars2 = ax2.barh(labels, percentages, color=colors2, edgecolor='black')
    ax2.set_xlabel('Percentage (%)', fontsize=12)
    ax2.set_title('By Percentage of True Class', fontsize=14)
    ax2.invert_yaxis()
    
    for i, (bar, pct) in enumerate(zip(bars2, percentages)):
        ax2.text(pct, bar.get_y() + bar.get_height()/2,
                f' {pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'top_confusion_pairs.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Top confusion pairs plot saved to {save_path}")
    
    plt.close()


def visualize_all(evaluator):
    """
    生成所有可视化图表
    
    Args:
        evaluator: Evaluator对象
    """
    print("\nGenerating all visualizations...")
    
    # 1. 训练曲线
    plot_training_curves()
    
    # 2. 混淆矩阵
    plot_confusion_matrix(evaluator.all_targets, evaluator.all_predictions, normalize=False)
    plot_confusion_matrix(evaluator.all_targets, evaluator.all_predictions, normalize=True)
    
    # 3. 每类准确率
    cm = evaluator.compute_confusion_matrix()
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plot_per_class_accuracy(per_class_acc)
    
    # 4. 误分类样本
    if len(evaluator.misclassified_samples) > 0:
        plot_misclassified_samples(
            evaluator.misclassified_samples,
            num_samples=min(20, len(evaluator.misclassified_samples))
        )
    
    # 5. 混淆对
    confusion_pairs = evaluator.get_confusion_pairs()
    if len(confusion_pairs) > 0:
        plot_top_confusion_pairs(confusion_pairs, top_k=10)
    
    print("\nAll visualizations completed!")


def create_analysis_report(evaluator, save_path=None):
    """
    创建误分类分析报告
    
    Args:
        evaluator: Evaluator对象
        save_path: 保存路径
    """
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'analysis.md')
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# CIFAR-10 分类模型误分类分析报告\n\n")
        
        # 1. 基本统计
        f.write("## 1. 基本统计信息\n\n")
        total_samples = len(evaluator.all_targets)
        correct_samples = sum([1 for t, p in zip(evaluator.all_targets, evaluator.all_predictions) if t == p])
        misclassified_samples = len(evaluator.misclassified_samples)
        
        f.write(f"- 总样本数: {total_samples}\n")
        f.write(f"- 正确分类: {correct_samples} ({correct_samples/total_samples*100:.2f}%)\n")
        f.write(f"- 误分类: {misclassified_samples} ({misclassified_samples/total_samples*100:.2f}%)\n\n")
        
        # 2. 每类准确率
        f.write("## 2. 每类准确率分析\n\n")
        cm = evaluator.compute_confusion_matrix()
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        f.write("| 类别 | 准确率 | 正确数/总数 |\n")
        f.write("|------|--------|-------------|\n")
        for i, class_name in enumerate(Config.CLASS_NAMES):
            correct = cm[i, i]
            total = cm[i].sum()
            f.write(f"| {class_name:12s} | {per_class_acc[i]*100:6.2f}% | {correct}/{total} |\n")
        f.write("\n")
        
        # 3. 最容易混淆的类别对
        f.write("## 3. 最容易混淆的类别对\n\n")
        confusion_pairs = evaluator.get_confusion_pairs()
        
        f.write("| 排名 | 真实类别 | 预测类别 | 数量 | 占比 |\n")
        f.write("|------|----------|----------|------|------|\n")
        for i, pair in enumerate(confusion_pairs[:10]):
            f.write(f"| {i+1} | {pair['true_class']:12s} | {pair['pred_class']:12s} | "
                   f"{pair['count']:3d} | {pair['percentage']:.1f}% |\n")
        f.write("\n")
        
        # 4. 误分类样本分析
        f.write("## 4. 典型误分类样本分析\n\n")
        analysis, sorted_samples = evaluator.analyze_misclassified_samples(num_samples=10)
        
        f.write("以下是模型最自信的错误预测（按预测置信度排序）:\n\n")
        for i, sample_info in enumerate(analysis['samples'][:10]):
            f.write(f"### 样本 {i+1}\n\n")
            f.write(f"- **真实类别**: {sample_info['true_class']}\n")
            f.write(f"- **预测类别**: {sample_info['predicted_class']}\n")
            f.write(f"- **预测置信度**: {sample_info['confidence']:.4f}\n")
            f.write(f"- **真实类别概率**: {sample_info['true_class_prob']:.4f}\n\n")
        
        # 5. 分析和建议
        f.write("## 5. 分析与改进建议\n\n")
        
        # 找出表现最差的类别
        worst_classes = np.argsort(per_class_acc)[:3]
        f.write("### 5.1 表现最差的类别\n\n")
        for i in worst_classes:
            f.write(f"- **{Config.CLASS_NAMES[i]}** (准确率: {per_class_acc[i]*100:.2f}%)\n")
        f.write("\n")
        
        # 主要混淆模式
        f.write("### 5.2 主要混淆模式\n\n")
        top_confusion = confusion_pairs[0]
        f.write(f"最常见的混淆是将 **{top_confusion['true_class']}** "
               f"误分类为 **{top_confusion['pred_class']}**，"
               f"共 {top_confusion['count']} 个样本 ({top_confusion['percentage']:.1f}%)。\n\n")
        
        f.write("### 5.3 改进建议\n\n")
        f.write("1. **数据增强**: 针对表现较差的类别增加数据增强策略\n")
        f.write("2. **类别平衡**: 检查训练数据是否存在类别不平衡问题\n")
        f.write("3. **特征学习**: 对易混淆的类别对，可以尝试:\n")
        f.write("   - 增加模型深度或复杂度\n")
        f.write("   - 使用注意力机制关注关键特征\n")
        f.write("   - 采用对比学习增强类别间的区分度\n")
        f.write("4. **损失函数**: 考虑使用Focal Loss等应对困难样本\n")
        f.write("5. **集成方法**: 使用模型集成来提高整体性能\n\n")
    
    print(f"Analysis report saved to {save_path}")


if __name__ == "__main__":
    from evaluate import Evaluator
    
    # 创建评估器并评估
    evaluator = Evaluator()
    evaluator.evaluate()
    
    # 生成所有可视化
    visualize_all(evaluator)
    
    # 生成分析报告
    create_analysis_report(evaluator)
