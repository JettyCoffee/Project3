"""
自动化超参数优化测试脚本
针对wide_resnet_small模型进行系统性的超参数搜索
测试不同的batch size、正则化系数、数据增强、训练技巧和损失函数
"""
import os
import sys
import json
import datetime
import itertools
import torch
from config import Config
from train import Trainer
from evaluate import evaluate_model
from visualize import visualize_all, create_analysis_report
import argparse


class HyperparameterSearcher:
    """超参数搜索器"""
    
    def __init__(self, base_results_dir='results', gpu_id=0):
        self.base_results_dir = base_results_dir
        self.gpu_id = gpu_id
        self.search_results = []
        
        # 创建搜索结果总目录
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        self.search_dir = os.path.join(base_results_dir, f'hypersearch_{timestamp}')
        os.makedirs(self.search_dir, exist_ok=True)
        
        print(f"超参数搜索结果将保存到: {self.search_dir}")
    
    def define_search_space(self):
        """定义搜索空间"""
        search_space = {
            # Batch size
            'batch_size': [64, 128, 256],
            
            # 正则化系数
            'weight_decay': [1e-4, 5e-4, 1e-3],
            
            # 数据增强配置
            'data_augmentation': [
                {'cutout': True, 'cutout_length': 12, 'random_flip': True},
                {'cutout': True, 'cutout_length': 16, 'random_flip': True},
                {'cutout': False, 'cutout_length': 0, 'random_flip': True},
            ],
            
            # 训练技巧
            'training_tricks': [
                {'mixup': False, 'label_smoothing': 0.0},
                {'mixup': True, 'mixup_alpha': 1.0, 'label_smoothing': 0.0},
                {'mixup': False, 'label_smoothing': 0.1},
            ],
            
            # 学习率调度
            'lr_scheduler': [
                {'type': 'cosine', 'lr': 0.1},
                {'type': 'multistep', 'lr': 0.1, 'milestones': [60, 120, 160]},
            ],
        }
        
        return search_space
    
    def generate_configs(self, search_space, max_configs=None):
        """生成所有配置组合"""
        # 生成所有组合
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        
        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)
        
        # 限制配置数量
        if max_configs and len(configs) > max_configs:
            import random
            random.shuffle(configs)
            configs = configs[:max_configs]
        
        return configs
    
    def apply_config(self, config_dict, experiment_name):
        """应用配置到Config类"""
        # 保存原始配置
        original_config = {}
        
        # 设置GPU
        Config.GPU_ID = self.gpu_id
        Config.DEVICE = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # 应用batch size
        original_config['BATCH_SIZE'] = Config.BATCH_SIZE
        Config.BATCH_SIZE = config_dict['batch_size']
        
        # 应用正则化系数
        original_config['WEIGHT_DECAY'] = Config.WEIGHT_DECAY
        Config.WEIGHT_DECAY = config_dict['weight_decay']
        
        # 应用数据增强
        aug_config = config_dict['data_augmentation']
        original_config['CUTOUT'] = Config.CUTOUT
        original_config['CUTOUT_LENGTH'] = Config.CUTOUT_LENGTH
        original_config['RANDOM_HORIZONTAL_FLIP'] = Config.RANDOM_HORIZONTAL_FLIP
        
        Config.CUTOUT = aug_config['cutout']
        Config.CUTOUT_LENGTH = aug_config['cutout_length']
        Config.RANDOM_HORIZONTAL_FLIP = aug_config['random_flip']
        
        # 应用训练技巧
        trick_config = config_dict['training_tricks']
        original_config['MIXUP'] = Config.MIXUP
        original_config['LABEL_SMOOTHING'] = Config.LABEL_SMOOTHING
        
        Config.MIXUP = trick_config['mixup']
        if 'mixup_alpha' in trick_config:
            original_config['MIXUP_ALPHA'] = Config.MIXUP_ALPHA
            Config.MIXUP_ALPHA = trick_config['mixup_alpha']
        Config.LABEL_SMOOTHING = trick_config['label_smoothing']
        
        # 应用学习率调度
        lr_config = config_dict['lr_scheduler']
        original_config['LR_SCHEDULER'] = Config.LR_SCHEDULER
        original_config['LEARNING_RATE'] = Config.LEARNING_RATE
        
        Config.LR_SCHEDULER = lr_config['type']
        Config.LEARNING_RATE = lr_config['lr']
        if 'milestones' in lr_config:
            original_config['LR_MILESTONES'] = Config.LR_MILESTONES
            Config.LR_MILESTONES = lr_config['milestones']
        
        # 设置实验目录
        experiment_dir = os.path.join(self.search_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 临时修改保存路径
        original_config['CHECKPOINT_DIR'] = Config.CHECKPOINT_DIR
        original_config['LOG_DIR'] = Config.LOG_DIR
        original_config['RESULTS_DIR'] = Config.RESULTS_DIR
        
        Config.CHECKPOINT_DIR = os.path.join(experiment_dir, 'checkpoints')
        Config.LOG_DIR = os.path.join(experiment_dir, 'logs')
        Config.RESULTS_DIR = experiment_dir
        
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        
        return original_config, experiment_dir
    
    def restore_config(self, original_config):
        """恢复原始配置"""
        for key, value in original_config.items():
            setattr(Config, key, value)
    
    def save_config_to_file(self, config_dict, experiment_dir):
        """保存配置到文件"""
        config_path = os.path.join(experiment_dir, 'hyperparameters.json')
        
        # 获取完整的Config配置
        full_config = {
            'search_config': config_dict,
            'full_config': Config.get_config_dict(),
            'timestamp': datetime.datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            full_config['cuda_version'] = torch.version.cuda
            full_config['gpu_id'] = Config.GPU_ID
            full_config['gpu_name'] = torch.cuda.get_device_name(Config.GPU_ID)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到: {config_path}")
    
    def run_experiment(self, config_dict, experiment_name):
        """运行单个实验"""
        print("\n" + "="*80)
        print(f"实验: {experiment_name}")
        print("="*80)
        
        # 应用配置
        original_config, experiment_dir = self.apply_config(config_dict, experiment_name)
        
        # 保存配置
        self.save_config_to_file(config_dict, experiment_dir)
        
        # 打印当前配置
        print("\n当前实验配置:")
        print(f"  Batch Size: {Config.BATCH_SIZE}")
        print(f"  Weight Decay: {Config.WEIGHT_DECAY}")
        print(f"  Cutout: {Config.CUTOUT} (length={Config.CUTOUT_LENGTH})")
        print(f"  Random Flip: {Config.RANDOM_HORIZONTAL_FLIP}")
        print(f"  Mixup: {Config.MIXUP}")
        print(f"  Label Smoothing: {Config.LABEL_SMOOTHING}")
        print(f"  LR Scheduler: {Config.LR_SCHEDULER}")
        print(f"  Learning Rate: {Config.LEARNING_RATE}")
        print(f"  Results Dir: {experiment_dir}")
        print()
        
        try:
            # 训练模型
            print("[1/3] 开始训练...")
            trainer = Trainer()
            history = trainer.train()
            
            # 评估模型
            print("\n[2/3] 开始评估...")
            best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            evaluator, results = evaluate_model(best_model_path)
            
            # 生成可视化
            print("\n[3/3] 生成可视化...")
            visualize_all(evaluator)
            create_analysis_report(evaluator)
            
            # 记录结果
            experiment_result = {
                'experiment_name': experiment_name,
                'config': config_dict,
                'best_val_acc': trainer.best_val_acc,
                'test_acc': results['overall_accuracy'],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'epochs_trained': len(history['train_loss']),
                'experiment_dir': experiment_dir
            }
            
            self.search_results.append(experiment_result)
            
            print(f"\n实验完成!")
            print(f"  最佳验证准确率: {trainer.best_val_acc:.2f}%")
            print(f"  测试准确率: {results['overall_accuracy']:.2f}%")
            
        except Exception as e:
            print(f"\n实验失败: {e}")
            import traceback
            traceback.print_exc()
            
            experiment_result = {
                'experiment_name': experiment_name,
                'config': config_dict,
                'error': str(e),
                'experiment_dir': experiment_dir
            }
            self.search_results.append(experiment_result)
        
        finally:
            # 恢复原始配置
            self.restore_config(original_config)
        
        return experiment_result
    
    def run_search(self, search_space, max_configs=None):
        """运行完整的超参数搜索"""
        print("\n" + "="*80)
        print("开始超参数搜索")
        print("="*80)
        
        # 生成所有配置
        configs = self.generate_configs(search_space, max_configs)
        print(f"\n总共需要测试 {len(configs)} 个配置")
        
        # 运行所有实验
        for idx, config in enumerate(configs, 1):
            experiment_name = f"exp_{idx:03d}"
            print(f"\n{'='*80}")
            print(f"进度: {idx}/{len(configs)}")
            print(f"{'='*80}")
            
            self.run_experiment(config, experiment_name)
        
        # 保存搜索摘要
        self.save_search_summary()
        
        # 打印最佳结果
        self.print_best_results()
    
    def save_search_summary(self):
        """保存搜索摘要"""
        summary_path = os.path.join(self.search_dir, 'search_summary.json')
        
        summary = {
            'total_experiments': len(self.search_results),
            'experiments': self.search_results,
            'search_completed': datetime.datetime.now().isoformat()
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n搜索摘要已保存到: {summary_path}")
    
    def print_best_results(self):
        """打印最佳结果"""
        print("\n" + "="*80)
        print("超参数搜索完成!")
        print("="*80)
        
        # 过滤掉失败的实验
        successful_results = [r for r in self.search_results if 'error' not in r]
        
        if not successful_results:
            print("没有成功的实验")
            return
        
        # 按测试准确率排序
        successful_results.sort(key=lambda x: x.get('test_acc', 0), reverse=True)
        
        print(f"\n成功完成的实验: {len(successful_results)}/{len(self.search_results)}")
        print("\nTop 5 最佳配置:")
        print("-"*80)
        
        for i, result in enumerate(successful_results[:5], 1):
            print(f"\n第 {i} 名:")
            print(f"  实验名称: {result['experiment_name']}")
            print(f"  测试准确率: {result['test_acc']:.2f}%")
            print(f"  验证准确率: {result['best_val_acc']:.2f}%")
            print(f"  配置:")
            config = result['config']
            print(f"    - Batch Size: {config['batch_size']}")
            print(f"    - Weight Decay: {config['weight_decay']}")
            print(f"    - Cutout: {config['data_augmentation']['cutout']} "
                  f"(length={config['data_augmentation']['cutout_length']})")
            print(f"    - Mixup: {config['training_tricks']['mixup']}")
            print(f"    - Label Smoothing: {config['training_tricks']['label_smoothing']}")
            print(f"    - LR Scheduler: {config['lr_scheduler']['type']}")
            print(f"  结果目录: {result['experiment_dir']}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Wide ResNet Small 超参数优化搜索')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='使用的GPU编号')
    parser.add_argument('--max-configs', type=int, default=None,
                       help='最多测试的配置数量（None表示测试所有）')
    parser.add_argument('--epochs', type=int, default=100,
                       help='每个实验的训练轮数')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（少量配置和轮数）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    import random
    import numpy as np
    seed = Config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置模型和基本参数
    Config.MODEL_NAME = 'wide_resnet_small'
    Config.NUM_EPOCHS = 10 if args.quick_test else args.epochs
    Config.PRETRAINED = False
    
    if args.quick_test:
        print("\n快速测试模式：减少训练轮数和配置数量")
        max_configs = 3
    else:
        max_configs = args.max_configs
    
    # 创建搜索器
    searcher = HyperparameterSearcher(gpu_id=args.gpu_id)
    
    # 定义搜索空间
    search_space = searcher.define_search_space()
    
    # 运行搜索
    searcher.run_search(search_space, max_configs=max_configs)


if __name__ == "__main__":
    main()
