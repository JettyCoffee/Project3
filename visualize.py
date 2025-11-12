"""
统一可视化脚本 - 合并训练曲线、混淆矩阵和Grad-CAM可视化
默认为所有10个类别生成Grad-CAM可视化
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix

from config import Config
from models import get_model
from data_loader import get_data_loaders
from grad_cam import GradCAM, visualize_gradcam_per_class, get_target_layer


def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('config', {})
    model_name = model_config.get('MODEL_NAME', 'resnet18')
    num_classes = model_config.get('NUM_CLASSES', 10)
    
    # 创建模型
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model_name}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    return model, model_name


def plot_training_curves(history_path=None, save_path=None):
    """绘制训练曲线（损失和准确率）"""
    if history_path is None:
        results_history_path = os.path.join(Config.RESULTS_DIR, 'training_history.json')
        if os.path.exists(results_history_path):
            history_path = results_history_path
        else:
            history_path = os.path.join(Config.LOG_DIR, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"History file {history_path} not found. Skipping training curves.")
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
    axes[0, 0].set_xticks(list(range(0, len(epochs)+1, 5)))

    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(list(range(0, len(epochs)+1, 5)))

    # 学习率曲线
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(list(range(0, len(epochs)+1, 5)))

    # 训练-验证准确率差异
    acc_diff = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, acc_diff, 'm-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference (%)')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(list(range(0, len(epochs)+1, 5)))
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    plt.close()


def evaluate_model(model, test_loader, device):
    """评估模型并返回预测结果"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    """绘制混淆矩阵"""
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
                xticklabels=class_names,
                yticklabels=class_names,
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


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """绘制每个类别的准确率柱状图"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(per_class_acc / per_class_acc.max())
    bars = plt.bar(class_names, per_class_acc * 100, color=colors, edgecolor='black')
    
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


def generate_gradcam_visualizations(model, model_name, test_loader, class_names, 
                                    save_dir, samples_per_class=1):
    """生成Grad-CAM可视化"""
    print("\n" + "="*60)
    print("Generating Grad-CAM Visualizations")
    print("="*60)
    
    # 获取目标层
    print(f"\nGetting target layer for {model_name}...")
    target_layer = get_target_layer(model, model_name)
    
    if target_layer is None:
        print("Error: Could not find target layer for Grad-CAM")
        return
    
    print(f"Target layer: {target_layer}")
    
    print(f"\nGenerating Grad-CAM...")
    visualize_gradcam_per_class(model, test_loader, class_names, target_layer,
                               save_dir=save_dir, samples_per_class=samples_per_class)
    
    print(f"\nGrad-CAM visualizations saved to {save_dir}")


def create_analysis_report(y_true, y_pred, class_names, save_path=None):
    """创建分析报告"""
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'analysis.md')
    
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# CIFAR-10 分类模型分析报告\n\n")
        
        f.write("## 1. 基本统计信息\n\n")
        total_samples = len(y_true)
        correct_samples = (y_true == y_pred).sum()
        misclassified_samples = total_samples - correct_samples
        
        f.write(f"- 总样本数: {total_samples}\n")
        f.write(f"- 正确分类: {correct_samples} ({correct_samples/total_samples*100:.2f}%)\n")
        f.write(f"- 误分类: {misclassified_samples} ({misclassified_samples/total_samples*100:.2f}%)\n\n")
        
        f.write("## 2. 每类准确率分析\n\n")
        f.write("| 类别 | 准确率 | 正确数/总数 |\n")
        f.write("|------|--------|-------------|\n")
        for i, class_name in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i].sum()
            f.write(f"| {class_name:12s} | {per_class_acc[i]*100:6.2f}% | {correct}/{total} |\n")
        f.write("\n")
        
        f.write("## 3. 最容易混淆的类别对\n\n")
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': class_names[i],
                        'pred_class': class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / cm[i].sum() * 100
                    })
        
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        f.write("| 排名 | 真实类别 | 预测类别 | 数量 | 占比 |\n")
        f.write("|------|----------|----------|------|------|\n")
        for i, pair in enumerate(confusion_pairs[:10]):
            f.write(f"| {i+1} | {pair['true_class']:12s} | {pair['pred_class']:12s} | "
                   f"{pair['count']:3d} | {pair['percentage']:.1f}% |\n")
        f.write("\n")
        
        f.write("## 4. 分析与改进建议\n\n")
        worst_classes = np.argsort(per_class_acc)[:3]
        f.write("### 4.1 表现最差的类别\n\n")
        for i in worst_classes:
            f.write(f"- **{class_names[i]}** (准确率: {per_class_acc[i]*100:.2f}%)\n")
        f.write("\n")
        
        if confusion_pairs:
            f.write("### 4.2 主要混淆模式\n\n")
            top_confusion = confusion_pairs[0]
            f.write(f"最常见的混淆是将 **{top_confusion['true_class']}** "
                   f"误分类为 **{top_confusion['pred_class']}**，"
                   f"共 {top_confusion['count']} 个样本 ({top_confusion['percentage']:.1f}%)。\n\n")
    
    print(f"Analysis report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified Visualization for CIFAR-10 Classification')
    
    # 基本参数
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--save-dir', type=str, default='results/visualizations',
                       help='保存目录')
    
    # Grad-CAM参数
    parser.add_argument('--samples-per-class', type=int, default=1,
                       help='每个类别的Grad-CAM样本数量（默认为1，即为所有10个类别生成）')
    parser.add_argument('--skip-gradcam', action='store_true',
                       help='跳过Grad-CAM可视化')
    
    # 其他可视化参数
    parser.add_argument('--skip-training-curves', action='store_true',
                       help='跳过训练曲线')
    parser.add_argument('--skip-confusion-matrix', action='store_true',
                       help='跳过混淆矩阵')
    parser.add_argument('--skip-per-class-acc', action='store_true',
                       help='跳过每类准确率图')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='跳过分析报告')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("CIFAR-10 Unified Visualization Tool")
    print("="*60)
    
    # 设置设备
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    # 加载模型
    print("\n" + "-"*60)
    print("Loading Model")
    print("-"*60)
    model, model_name = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载数据
    print("\n" + "-"*60)
    print("Loading Data")
    print("-"*60)
    _, _, test_loader = get_data_loaders()
    
    # 1. 训练曲线
    if not args.skip_training_curves:
        print("\n" + "-"*60)
        print("1. Training Curves")
        print("-"*60)
        plot_training_curves(save_path=os.path.join(args.save_dir, 'training_curves.png'))
    
    # 评估模型（用于混淆矩阵和每类准确率）
    if not (args.skip_confusion_matrix and args.skip_per_class_acc and args.skip_analysis):
        print("\n" + "-"*60)
        print("Model Evaluation")
        print("-"*60)
        y_true, y_pred = evaluate_model(model, test_loader, device)
        accuracy = (y_true == y_pred).mean() * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # 2. 混淆矩阵
    if not args.skip_confusion_matrix:
        print("\n" + "-"*60)
        print("2. Confusion Matrix")
        print("-"*60)
        plot_confusion_matrix(y_true, y_pred, Config.CLASS_NAMES, 
                            save_path=os.path.join(args.save_dir, 'confusion_matrix.png'))
        plot_confusion_matrix(y_true, y_pred, Config.CLASS_NAMES, 
                            save_path=os.path.join(args.save_dir, 'confusion_matrix_normalized.png'),
                            normalize=True)
    
    # 3. 每类准确率
    if not args.skip_per_class_acc:
        print("\n" + "-"*60)
        print("3. Per-Class Accuracy")
        print("-"*60)
        plot_per_class_accuracy(y_true, y_pred, Config.CLASS_NAMES,
                               save_path=os.path.join(args.save_dir, 'per_class_accuracy.png'))
    
    # 4. Grad-CAM可视化
    if not args.skip_gradcam:
        print("\n" + "-"*60)
        print("4. Grad-CAM Visualizations (All 10 Classes)")
        print("-"*60)
        gradcam_dir = os.path.join(args.save_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        generate_gradcam_visualizations(model, model_name, test_loader, Config.CLASS_NAMES,
                                       gradcam_dir, args.samples_per_class)
    
    # 5. 分析报告
    if not args.skip_analysis:
        print("\n" + "-"*60)
        print("5. Analysis Report")
        print("-"*60)
        create_analysis_report(y_true, y_pred, Config.CLASS_NAMES,
                              save_path=os.path.join(args.save_dir, 'analysis.md'))
    
    # 总结
    print("\n" + "="*60)
    print("All Visualizations Completed!")
    print("="*60)
    print(f"\nResults saved in: {args.save_dir}")
    print("\nGenerated files:")
    if not args.skip_training_curves:
        print("training_curves.png")
    if not args.skip_confusion_matrix:
        print("confusion_matrix.png")
        print("confusion_matrix_normalized.png")
    if not args.skip_per_class_acc:
        print("per_class_accuracy.png")
    if not args.skip_gradcam:
        print("gradcam/gradcam_all_classes.png")
    if not args.skip_analysis:
        print("analysis.md")
    print("="*60)


if __name__ == "__main__":
    main()
