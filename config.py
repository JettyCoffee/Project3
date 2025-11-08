"""
配置文件 - 统一管理所有超参数和路径
"""
import torch
import os

class Config:
    """项目配置类"""
    
    # 项目路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    # 设备配置
    DEVICE = torch.device('cuda')
    NUM_WORKERS = 8
    
    # 数据集配置
    DATASET_NAME = 'CIFAR10'
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    
    # CIFAR-10 类别名称
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 数据划分
    TRAIN_RATIO = 0.9  # 训练集占比
    VAL_RATIO = 0.1    # 验证集占比

    # 训练超参数
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1 # 初始学习率
    MOMENTUM = 0.9 # 动量因子
    WEIGHT_DECAY = 5e-4 # L2正则化系数
    
    # 学习率调度
    LR_SCHEDULER = 'cosine'  # 'cosine', 'step', 'multistep'
    LR_STEP_SIZE = 40 # cosine不需要该参数
    LR_GAMMA = 0.1 # 
    LR_MILESTONES = [60, 120, 160] # 多步调度的里程碑
    
    # 早停配置
    EARLY_STOPPING = True
    PATIENCE = 15  # 在验证集上多少个epoch没有提升就停止
    MIN_DELTA = 0.001  # 最小改善量
    
    # 模型选择
    MODEL_NAME = 'wide_resnet_small'  # 'resnet18', 'resnet34', 'resnet50', 'wide_resnet', 'wide_resnet_small', 'dla34', 'vit', 'vit_small'
    PRETRAINED = True  # 是否使用预训练权重
    
    # 数据增强配置
    DATA_AUGMENTATION = True
    RANDOM_CROP = True
    RANDOM_HORIZONTAL_FLIP = True
    CUTOUT = True
    CUTOUT_LENGTH = 16
    
    # 正则化配置
    USE_DROPOUT = True
    DROPOUT_RATE = 0.5
    USE_BATCH_NORM = True
    
    # 训练技巧
    MIXUP = False  # Mixup数据增强
    MIXUP_ALPHA = 1.0
    LABEL_SMOOTHING = 0.1  # 标签平滑
    
    # 损失函数配置
    LOSS_FUNCTION = 'cross_entropy'  # 'cross_entropy', 'focal', 'label_smoothing', 'combined'
    FOCAL_GAMMA = 2.0  # Focal Loss的gamma参数
    FOCAL_ALPHA = None  # Focal Loss的alpha参数（类别权重）
    SMOOTHING_EPSILON = 0.1  # Label Smoothing的epsilon参数
    
    # 日志和保存
    SAVE_FREQ = 10  # 每隔多少个epoch保存一次模型
    LOG_FREQ = 100  # 每隔多少个batch打印一次日志
    SAVE_BEST_ONLY = True  # 只保存最佳模型
    
    # 评估配置
    EVAL_BATCH_SIZE = 100
    TOP_K_ACCURACY = [1, 3, 5]  # 计算Top-K准确率
    
    # 可视化配置
    PLOT_MISCLASSIFIED = True
    NUM_MISCLASSIFIED_SAMPLES = 20  # 展示的误分类样本数量
    SAVE_CONFUSION_MATRIX = True
    SAVE_TRAINING_CURVES = True
    
    # Grad-CAM可视化配置
    ENABLE_GRADCAM = True  # 是否启用Grad-CAM可视化
    GRADCAM_NUM_SAMPLES = 9  # Grad-CAM可视化的样本数量
    GRADCAM_SAVE_DIR = 'results'  # Grad-CAM保存目录
    
    # 种子设置（用于结果可复现）
    SEED = 42
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典（用于日志记录，只返回可序列化的值）"""
        config_dict = {}
        for k, v in cls.__dict__.items():
            # 排除私有属性、方法、类方法、静态方法等
            if k.startswith('_'):
                continue
            # 排除方法和类方法
            if callable(v) or isinstance(v, (classmethod, staticmethod)):
                continue
            # 排除类型对象
            if isinstance(v, type):
                continue
            
            # 转换torch.device为字符串以支持序列化
            if isinstance(v, torch.device):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
        return config_dict
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)
        config_dict = cls.get_config_dict()
        for key, value in config_dict.items():
            if not key.startswith('__'):
                print(f"{key:30s}: {value}")
        print("=" * 50)
