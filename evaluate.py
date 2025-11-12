"""
评估模块 - 实现测试集评估、混淆矩阵生成和误分类分析
"""
import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from config import Config
from data_loader import get_data_loaders
from models import get_model


class Evaluator:
    """评估器类"""
    def __init__(self, checkpoint_path=None):
        self.device = Config.DEVICE
        self.config = Config
        
        # 数据加载
        print("Loading data...")
        _, _, self.test_loader = get_data_loaders()
        
        # 模型初始化
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = get_model(
            Config.MODEL_NAME,
            num_classes=Config.NUM_CLASSES,
            pretrained=False
        ).to(self.device)
        
        # 加载模型权重
        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
        
        self.checkpoint = None
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            print(f"Model loaded. Best validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A')}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        
        self.model.eval()
        
        # 用于存储结果
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
        self.misclassified_samples = []
    
    def evaluate(self):
        """在测试集上评估模型"""
        print("\nEvaluating on test set...")
        
        self.all_predictions = []
        self.all_targets = []
        self.all_probs = []
        self.misclassified_samples = []
        
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
                # 获取概率和预测
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # 记录结果
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_targets.extend(targets.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
                
                # 统计准确率
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 记录误分类样本
                misclassified_mask = ~predicted.eq(targets)
                if misclassified_mask.any():
                    misclassified_indices = torch.where(misclassified_mask)[0]
                    for idx in misclassified_indices:
                        self.misclassified_samples.append({
                            'image': inputs[idx].cpu(),
                            'true_label': targets[idx].item(),
                            'pred_label': predicted[idx].item(),
                            'probs': probs[idx].cpu().numpy(),
                            'batch_idx': batch_idx,
                            'sample_idx': idx.item()
                        })
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # 计算最终指标
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  Correct: {correct}/{total}")
        print(f"  Misclassified: {len(self.misclassified_samples)}")
        
        return test_loss, test_acc
    
    def compute_confusion_matrix(self):
        """计算混淆矩阵"""
        cm = confusion_matrix(self.all_targets, self.all_predictions)
        return cm
    
    def compute_per_class_accuracy(self):
        """计算每个类别的准确率"""
        cm = self.compute_confusion_matrix()
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        print("\nPer-class Accuracy:")
        print("-" * 40)
        for i, class_name in enumerate(Config.CLASS_NAMES):
            print(f"{class_name:15s}: {per_class_acc[i]*100:.2f}%")
        print("-" * 40)
        
        return per_class_acc
    
    def get_classification_report(self):
        """获取详细的分类报告"""
        report = classification_report(
            self.all_targets,
            self.all_predictions,
            target_names=Config.CLASS_NAMES,
            digits=4
        )
        print("\nClassification Report:")
        print(report)
        return report
    
    def compute_topk_accuracy(self, k=5):
        """计算Top-K准确率"""
        correct_topk = 0
        probs_array = np.array(self.all_probs)
        targets_array = np.array(self.all_targets)
        
        # 获取top-k预测
        topk_predictions = np.argsort(probs_array, axis=1)[:, -k:]
        
        # 检查真实标签是否在top-k中
        for i, target in enumerate(targets_array):
            if target in topk_predictions[i]:
                correct_topk += 1
        
        topk_acc = 100. * correct_topk / len(targets_array)
        print(f"Top-{k} Accuracy: {topk_acc:.2f}%")
        return topk_acc
    
    def analyze_misclassified_samples(self, num_samples=20):
        """分析误分类样本"""
        print(f"\nAnalyzing {min(num_samples, len(self.misclassified_samples))} misclassified samples...")
        
        analysis = {
            'total_misclassified': len(self.misclassified_samples),
            'samples': []
        }
        
        # 按置信度排序（最自信的错误预测）
        sorted_samples = sorted(
            self.misclassified_samples,
            key=lambda x: x['probs'][x['pred_label']],
            reverse=True
        )
        
        for i, sample in enumerate(sorted_samples[:num_samples]):
            true_label = sample['true_label']
            pred_label = sample['pred_label']
            confidence = sample['probs'][pred_label]
            
            sample_info = {
                'index': i,
                'true_class': Config.CLASS_NAMES[true_label],
                'predicted_class': Config.CLASS_NAMES[pred_label],
                'confidence': float(confidence),
                'true_class_prob': float(sample['probs'][true_label])
            }
            
            analysis['samples'].append(sample_info)
            
            if i < 10:  # 打印前10个
                print(f"\nSample {i+1}:")
                print(f"  True: {Config.CLASS_NAMES[true_label]:12s} (prob: {sample['probs'][true_label]:.4f})")
                print(f"  Pred: {Config.CLASS_NAMES[pred_label]:12s} (prob: {confidence:.4f})")
        
        return analysis, sorted_samples[:num_samples]
    
    def get_confusion_pairs(self):
        """获取最容易混淆的类别对"""
        cm = self.compute_confusion_matrix()
        
        # 找出非对角线上的最大值（最容易混淆的类别对）
        confusion_pairs = []
        for i in range(len(Config.CLASS_NAMES)):
            for j in range(len(Config.CLASS_NAMES)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': Config.CLASS_NAMES[i],
                        'pred_class': Config.CLASS_NAMES[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        # 按数量排序
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nTop 10 Confusion Pairs:")
        print("-" * 60)
        for i, pair in enumerate(confusion_pairs[:10]):
            print(f"{i+1}. {pair['true_class']:12s} → {pair['pred_class']:12s}: "
                  f"{pair['count']:3d} ({pair['percentage']:.1f}%)")
        print("-" * 60)
        
        return confusion_pairs
    
    def save_results(self):
        """保存评估结果"""
        results = {
            'test_accuracy': accuracy_score(self.all_targets, self.all_predictions) * 100,
            'per_class_accuracy': {},
            'confusion_pairs': [],
            'topk_accuracy': {}
        }
        
        # 每类准确率
        per_class_acc = self.compute_per_class_accuracy()
        for i, class_name in enumerate(Config.CLASS_NAMES):
            results['per_class_accuracy'][class_name] = float(per_class_acc[i] * 100)
        
        # Top-K准确率
        for k in Config.TOP_K_ACCURACY:
            topk_acc = self.compute_topk_accuracy(k)
            results['topk_accuracy'][f'top_{k}'] = topk_acc
        
        # 混淆对
        results['confusion_pairs'] = self.get_confusion_pairs()[:10]
        
        # 保存到JSON
        results_path = os.path.join(Config.RESULTS_DIR, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")
        
        # 保存分类报告
        report = self.get_classification_report()
        report_path = os.path.join(Config.RESULTS_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")
        
        return results


def evaluate_model(checkpoint_path=None):
    """评估模型的主函数"""
    # 创建结果目录
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # 创建评估器
    evaluator = Evaluator(checkpoint_path)
    
    # 评估
    test_loss, test_acc = evaluator.evaluate()
    
    # 详细分析
    evaluator.compute_per_class_accuracy()
    evaluator.get_classification_report()
    
    # Top-K准确率
    for k in Config.TOP_K_ACCURACY:
        evaluator.compute_topk_accuracy(k)
    
    # 误分类分析
    analysis, misclassified_samples = evaluator.analyze_misclassified_samples(
        num_samples=Config.NUM_MISCLASSIFIED_SAMPLES
    )
    
    # 混淆对分析
    evaluator.get_confusion_pairs()
    
    # 保存结果
    results = evaluator.save_results()
    
    return evaluator, results


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # 评估
    evaluator, results = evaluate_model()
