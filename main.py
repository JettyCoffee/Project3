"""
主程序入口 - 整合训练、评估和可视化流程
支持命令行参数配置
"""
import argparse
import os
import sys
import torch
import numpy as np
import random
import datetime
import shutil

from config import Config
from train import Trainer
from evaluate import evaluate_model
from visualize import visualize_all, create_analysis_report


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories():
    """创建必要的目录"""
    directories = [
        Config.DATA_DIR,
        Config.CHECKPOINT_DIR,
        Config.LOG_DIR,
        Config.RESULTS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directories created:")
    for directory in directories:
        print(f"  - {directory}")


def rename_checkpoint_with_timestamp():
    """训练完成后为检查点添加时间戳"""
    checkpoint_dir = Config.CHECKPOINT_DIR
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    # 重命名best_model.pth
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        new_best_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
        shutil.move(best_model_path, new_best_path)
        print(f"Best model renamed to: {new_best_path}")
    
    # 重命名last_checkpoint.pth
    last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint_path):
        new_last_path = os.path.join(checkpoint_dir, f'last_checkpoint_{timestamp}.pth')
        shutil.move(last_checkpoint_path, new_last_path)
        print(f"Last checkpoint renamed to: {new_last_path}")
    
    return timestamp


def create_timestamped_results_dir():
    """创建带时间戳的结果目录"""
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    timestamped_dir = os.path.join(Config.RESULTS_DIR, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir, timestamp


def save_hyperparameters(save_dir, args):
    """保存当前测试的所有超参数到JSON文件"""
    import json
    
    # 获取当前配置
    config_dict = Config.get_config_dict()
    
    # 添加命令行参数
    hyperparams = {
        'config': config_dict,
        'command_line_args': {
            'mode': args.mode,
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'resume': args.resume,
            'checkpoint': args.checkpoint,
            'visualize': args.visualize,
            'seed': args.seed,
            'device': str(args.device) if args.device else None
        },
        'timestamp': datetime.datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        hyperparams['cuda_version'] = torch.version.cuda
        hyperparams['gpu_name'] = torch.cuda.get_device_name(0)
    
    # 保存到JSON文件
    hyperparams_path = os.path.join(save_dir, 'hyperparameters.json')
    with open(hyperparams_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)
    
    print(f"Hyperparameters saved to: {hyperparams_path}")
    return hyperparams_path


def train_mode(args):
    """训练模式"""
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # 更新配置
    if args.model:
        Config.MODEL_NAME = args.model
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # 创建训练器
    trainer = Trainer()
    
    # 如果指定了恢复训练
    if args.resume:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'last_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Resume checkpoint not found at {checkpoint_path}")
    
    # 开始训练
    history = trainer.train()
    
    # 获取训练器的run_id（包含模型名和时间戳）
    run_id = trainer.run_id
    
    print(f"\nTraining completed successfully! Run ID: {run_id}")
    print(f"Best model saved at: checkpoints/best_model_{run_id}.pth")
    return history, run_id


def evaluate_mode(args, custom_results_dir=None):
    """评估模式"""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # 创建带时间戳的结果目录（如果没有提供自定义目录）
    if custom_results_dir is None:
        timestamped_dir, timestamp = create_timestamped_results_dir()
        print(f"Results will be saved to: {timestamped_dir}")
    else:
        timestamped_dir = custom_results_dir
        timestamp = os.path.basename(custom_results_dir)
    
    # 临时更改Config.RESULTS_DIR
    original_results_dir = Config.RESULTS_DIR
    Config.RESULTS_DIR = timestamped_dir
    
    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    
    try:
        # 复制训练历史文件到结果目录（如果存在）
        history_path = os.path.join(Config.LOG_DIR, 'training_history.json')
        if os.path.exists(history_path):
            target_history_path = os.path.join(timestamped_dir, 'training_history.json')
            shutil.copy2(history_path, target_history_path)
            print(f"Training history copied to: {target_history_path}")
        
        # 复制最佳模型到结果目录
        if os.path.exists(checkpoint_path):
            target_model_path = os.path.join(timestamped_dir, os.path.basename(checkpoint_path))
            shutil.copy2(checkpoint_path, target_model_path)
            print(f"Model checkpoint copied to: {target_model_path}")
        
        # 评估
        evaluator, results = evaluate_model(checkpoint_path)
        
        # 生成可视化
        if args.visualize:
            print("\nGenerating visualizations...")
            visualize_all(evaluator)
            create_analysis_report(evaluator)
        
        # 保存超参数信息
        save_hyperparameters(timestamped_dir, args)
        
        print("\nEvaluation completed successfully!")
        print(f"All results saved in: {timestamped_dir}")
        
        # 恢复原始结果目录
        Config.RESULTS_DIR = original_results_dir
            
        return evaluator, results, timestamp
        
    except Exception as e:
        # 恢复原始结果目录
        Config.RESULTS_DIR = original_results_dir
        raise e


def visualize_mode(args):
    """可视化模式"""
    print("\n" + "="*60)
    print("VISUALIZATION MODE")
    print("="*60)
    
    from evaluate import Evaluator
    
    # 创建带时间戳的结果目录
    timestamped_dir, timestamp = create_timestamped_results_dir()
    print(f"Results will be saved to: {timestamped_dir}")
    
    # 临时更改Config.RESULTS_DIR
    original_results_dir = Config.RESULTS_DIR
    Config.RESULTS_DIR = timestamped_dir
    
    try:
        # 复制训练历史文件到结果目录（如果存在）
        history_path = os.path.join(Config.LOG_DIR, 'training_history.json')
        if os.path.exists(history_path):
            target_history_path = os.path.join(timestamped_dir, 'training_history.json')
            shutil.copy2(history_path, target_history_path)
            print(f"Training history copied to: {target_history_path}")
        
        # 创建评估器
        checkpoint_path = args.checkpoint if args.checkpoint else None
        if checkpoint_path and os.path.exists(checkpoint_path):
            target_model_path = os.path.join(timestamped_dir, os.path.basename(checkpoint_path))
            shutil.copy2(checkpoint_path, target_model_path)
            print(f"Model checkpoint copied to: {target_model_path}")
        
        evaluator = Evaluator(checkpoint_path)
        
        # 评估（生成数据）
        evaluator.evaluate()
        
        # 生成可视化
        visualize_all(evaluator)
        create_analysis_report(evaluator)
        
        # 保存超参数信息
        save_hyperparameters(timestamped_dir, args)
        
        print("\nVisualization completed successfully!")
        print(f"All results saved in: {timestamped_dir}")
        
        # 恢复原始结果目录
        Config.RESULTS_DIR = original_results_dir
        
        return timestamp
        
    except Exception as e:
        # 恢复原始结果目录
        Config.RESULTS_DIR = original_results_dir
        raise e


def full_pipeline(args):
    """完整流程：训练 + 评估 + 可视化"""
    print("\n" + "="*60)
    print("FULL PIPELINE MODE")
    print("="*60)
    
    # 1. 训练
    print("\n[Step 1/3] Training...")
    history, run_id = train_mode(args)
    
    # 2. 创建带时间戳的结果目录并评估
    print("\n[Step 2/3] Evaluating...")
    timestamped_dir, eval_timestamp = create_timestamped_results_dir()
    
    # 使用训练后的最新检查点（使用run_id）
    args.checkpoint = os.path.join(Config.CHECKPOINT_DIR, f'best_model_{run_id}.pth')
    args.visualize = True
    
    evaluator, results, _ = evaluate_mode(args, custom_results_dir=timestamped_dir)
    
    # 3. 完成
    print("\n[Step 3/3] All done!")
    print("\nFull pipeline completed successfully!")
    print(f"Best model saved at: {args.checkpoint}")
    print(f"Results saved in: {timestamped_dir}")
    
    return history, evaluator, results, eval_timestamp


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CIFAR-10 Image Classification with CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'eval', 'visualize', 'full'],
                       help='运行模式: train/eval/visualize/full')
    
    # 训练相关参数
    parser.add_argument('--model', type=str, default=None,
                       choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet', 
                               'wide_resnet_small', 'dla34', 'vit'],
                       help='模型架构')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--resume', action='store_true',
                       help='从上次检查点恢复训练')
    
    # 评估相关参数
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--visualize', action='store_true',
                       help='在评估后生成可视化')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=Config.SEED,
                       help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='GPU编号 (例如: 0, 1, 2...)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.gpu_id is not None:
        Config.GPU_ID = args.gpu_id
        Config.DEVICE = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    elif args.device:
        if args.device == 'cuda':
            Config.DEVICE = torch.device(f'cuda:{Config.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        else:
            Config.DEVICE = torch.device(args.device)
    
    # 创建必要目录
    create_directories()
    
    # 打印系统信息
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU ID: {Config.GPU_ID}")
        print(f"GPU: {torch.cuda.get_device_name(Config.GPU_ID)}")
    print(f"Device: {Config.DEVICE}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # 根据模式运行
    try:
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'eval':
            evaluate_mode(args)
        elif args.mode == 'visualize':
            visualize_mode(args)
        elif args.mode == 'full':
            full_pipeline(args)
        
        print("\n" + "="*60)
        print("PROGRAM COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
