"""
模型定义模块 - 实现多种CNN架构
包括：自定义CNN、ResNet、VGG、MobileNetV2等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import Config


class CustomCNN(nn.Module):
    """
    自定义CNN模型 - 经典的卷积神经网络架构
    适合CIFAR-10数据集的小尺寸图像
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # 第三个卷积块
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """ResNet的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 模型（适配CIFAR-10）
    相比原始ResNet，第一层不使用7x7卷积和最大池化
    """
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        # 初始层（适配小图像）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class VGG16(nn.Module):
    """
    VGG-16 模型（适配CIFAR-10）
    精简版本，适合小尺寸图像
    """
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name='custom_cnn', num_classes=10, pretrained=False):
    """
    根据名称获取模型
    
    Args:
        model_name (str): 模型名称
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重
        
    Returns:
        nn.Module: PyTorch模型
    """
    model_name = model_name.lower()
    
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)
        
    elif model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(pretrained=True)
            # 修改第一层以适应CIFAR-10
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            # 修改最后的全连接层
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = ResNet18(num_classes=num_classes)
            
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vgg16':
        if pretrained:
            model = models.vgg16_bn(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        else:
            model = VGG16(num_classes=num_classes)
            
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试所有模型
    print("Testing models...")
    
    models_to_test = ['custom_cnn', 'resnet18', 'vgg16']
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print('='*50)
        
        model = get_model(model_name, num_classes=10, pretrained=False)
        
        # 统计参数
        params = count_parameters(model)
        print(f"Total parameters: {params:,}")
        
        # 测试前向传播
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
