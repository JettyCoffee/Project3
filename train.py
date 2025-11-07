"""
训练模块 - 实现模型训练、验证、学习率调度和早停
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
import time
import os
import json
from tqdm import tqdm

from config import Config
from data_loader import get_data_loaders, mixup_data, mixup_criterion
from models import get_model, count_parameters


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        """
        Args:
            patience (int): 容忍的epoch数
            min_delta (float): 最小改善量
            mode (str): 'max' 或 'min'，表示指标越大越好还是越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class Trainer:
    """训练器类"""
    def __init__(self):
        self.device = Config.DEVICE
        self.config = Config
        
        # 数据加载
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders()
        
        # 模型初始化
        print(f"\nInitializing model: {Config.MODEL_NAME}")
        self.model = get_model(
            Config.MODEL_NAME,
            num_classes=Config.NUM_CLASSES,
            pretrained=Config.PRETRAINED
        ).to(self.device)
        
        print(f"Total parameters: {count_parameters(self.model):,}")
        
        # 损失函数
        if Config.LABEL_SMOOTHING > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 早停
        if Config.EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=Config.PATIENCE,
                min_delta=Config.MIN_DELTA,
                mode='max'
            )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
    def _get_scheduler(self):
        """获取学习率调度器"""
        if Config.LR_SCHEDULER == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=Config.NUM_EPOCHS)
        elif Config.LR_SCHEDULER == 'step':
            return StepLR(self.optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
        elif Config.LR_SCHEDULER == 'multistep':
            return MultiStepLR(self.optimizer, milestones=Config.LR_MILESTONES, gamma=Config.LR_GAMMA)
        else:
            return None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixup数据增强
            if Config.MIXUP:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, Config.MIXUP_ALPHA)
                
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            if Config.MIXUP:
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if Config.MIXUP:
                correct += (lam * predicted.eq(targets_a).sum().item() +
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            if (batch_idx + 1) % Config.LOG_FREQ == 0 or (batch_idx + 1) == len(self.train_loader):
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]  ')
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        # 使用Config类提供的方法获取可序列化的配置字典
        config_dict = Config.get_config_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': config_dict
        }
        
        # 保存最新的检查点
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found.")
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.start_epoch}, best val acc: {self.best_val_acc:.2f}%")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        Config.print_config()
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
                # 如果学习率低于0.001，固定为0.001
                new_lr = self.optimizer.param_groups[0]['lr']
            #    if new_lr < 0.01:
            #        for param_group in self.optimizer.param_groups:
            #            param_group['lr'] = 0.01
            #        print(f"  Learning rate adjusted: {new_lr:.6f} -> 0.001000 (minimum threshold)")
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 打印总结
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # 保存检查点
            if (epoch + 1) % Config.SAVE_FREQ == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
            
            # 早停检查
            if Config.EARLY_STOPPING:
                if self.early_stopping(val_acc):
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            print("-" * 50)
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 保存训练历史
        history_path = os.path.join(Config.LOG_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")
        
        return self.history


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # 创建必要的目录
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 初始化训练器并训练
    trainer = Trainer()
    history = trainer.train()
