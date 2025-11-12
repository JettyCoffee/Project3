"""
数据加载模块 - 实现CIFAR-10数据集加载、预处理和增强
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from config import Config


class Cutout:
    """Cutout数据增强：随机遮挡图像的一个矩形区域"""
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): 形状为 (C, H, W) 的张量
        Returns:
            Tensor: 应用Cutout后的图像
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


def get_transforms(train=True):
    """
    获取数据预处理和增强的transforms
    
    Args:
        train (bool): 是否为训练集
        
    Returns:
        transforms.Compose: 组合的变换操作
    """
    if train and Config.DATA_AUGMENTATION:
        transform_list = []
        
        # 随机裁剪和填充
        if Config.RANDOM_CROP:
            transform_list.append(transforms.RandomCrop(32, padding=4))
        
        # 随机水平翻转
        if Config.RANDOM_HORIZONTAL_FLIP:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # 转换为张量并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Cutout增强
        if Config.CUTOUT:
            transform_list.append(Cutout(Config.CUTOUT_LENGTH))
        
        return transforms.Compose(transform_list)
    else:
        # 测试/验证集只进行标准化
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])


def get_data_loaders():
    """
    获取训练集、验证集和测试集的DataLoader
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 设置随机种子以保证可重复性
    torch.manual_seed(Config.SEED)
    
    # 下载并加载训练数据
    print("Loading CIFAR-10 dataset...")
    train_transform = get_transforms(train=True)
    val_test_transform = get_transforms(train=False)
    
    # 完整训练集
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=False,
        download=True,
        transform=val_test_transform
    )
    
    # 划分训练集和验证集
    train_size = int(Config.TRAIN_RATIO * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # 为验证集创建单独的数据集（使用测试集的transform）
    val_dataset_with_test_transform = torchvision.datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=True,
        download=False,
        transform=val_test_transform
    )
    # 使用相同的索引
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_with_test_transform, val_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset):
    """
    获取数据集的类别分布
    
    Args:
        dataset: PyTorch数据集
        
    Returns:
        dict: 类别名称到样本数量的映射
    """
    class_counts = {}
    for _, label in dataset:
        class_name = Config.CLASS_NAMES[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts


def mixup_data(x, y, alpha=1.0):
    """
    Mixup数据增强
    
    Args:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup损失函数
    
    Args:
        criterion: 损失函数
        pred: 模型预测
        y_a, y_b: 混合的两个标签
        lam: 混合比例
        
    Returns:
        float: 混合后的损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    # 测试数据加载
    print("Testing data loader...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 查看一个batch的数据
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Min pixel value: {images.min():.4f}")
    print(f"Max pixel value: {images.max():.4f}")
    print(f"Mean: {images.mean():.4f}")
    print(f"Std: {images.std():.4f}")
