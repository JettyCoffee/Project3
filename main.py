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
    
    print("\nTraining completed successfully!")
    return history


def evaluate_mode(args):
    """评估模式"""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    
    # 评估
    evaluator, results = evaluate_model(checkpoint_path)
    
    # 生成可视化
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_all(evaluator)
        create_analysis_report(evaluator)
    
    print("\nEvaluation completed successfully!")
    return evaluator, results


def visualize_mode(args):
    """可视化模式"""
    print("\n" + "="*60)
    print("VISUALIZATION MODE")
    print("="*60)
    
    from evaluate import Evaluator
    
    # 创建评估器
    checkpoint_path = args.checkpoint if args.checkpoint else None
    evaluator = Evaluator(checkpoint_path)
    
    # 评估（生成数据）
    evaluator.evaluate()
    
    # 生成可视化
    visualize_all(evaluator)
    create_analysis_report(evaluator)
    
    print("\nVisualization completed successfully!")


def full_pipeline(args):
    """完整流程：训练 + 评估 + 可视化"""
    print("\n" + "="*60)
    print("FULL PIPELINE MODE")
    print("="*60)
    
    # 1. 训练
    print("\n[Step 1/3] Training...")
    history = train_mode(args)
    
    # 2. 评估
    print("\n[Step 2/3] Evaluating...")
    args.checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    args.visualize = True
    evaluator, results = evaluate_mode(args)
    
    # 3. 完成
    print("\n[Step 3/3] All done!")
    print("\nFull pipeline completed successfully!")
    print(f"Best model saved at: {args.checkpoint}")
    print(f"Results saved in: {Config.RESULTS_DIR}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CIFAR-10 Image Classification with CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 训练模型
  python main.py --mode train --model resnet18 --epochs 100

  # 评估模型
  python main.py --mode eval --checkpoint checkpoints/best_model.pth

  # 只生成可视化
  python main.py --mode visualize

  # 完整流程（训练+评估+可视化）
  python main.py --mode full --model resnet18 --epochs 50
        """
    )
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'eval', 'visualize', 'full'],
                       help='运行模式: train/eval/visualize/full')
    
    # 训练相关参数
    parser.add_argument('--model', type=str, default=None,
                       choices=['custom_cnn', 'resnet18', 'resnet34', 'vgg16', 'mobilenetv2'],
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
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device:
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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
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
