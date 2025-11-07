"""
Grad-CAM可视化模块 - 展示模型关注的区域
包括：Grad-CAM、特征图可视化等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    论文：Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: 要可视化的模型
            target_layer: 目标卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.hook_handles = []
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 注册钩子
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        self.hook_handles.append(forward_handle)
        self.hook_handles.append(backward_handle)
    
    def remove_hooks(self):
        """移除钩子"""
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """
        生成类激活图
        
        Args:
            input_image: 输入图像，shape为(1, C, H, W)
            target_class: 目标类别，如果为None则使用预测类别
            
        Returns:
            cam: 类激活图
            predicted_class: 预测类别
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_image)
        
        # 获取预测类别
        predicted_class = output.argmax(dim=1).item()
        
        if target_class is None:
            target_class = predicted_class
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, predicted_class
    
    def __del__(self):
        """析构函数，自动移除钩子"""
        self.remove_hooks()


class FeatureMapVisualizer:
    """特征图可视化器"""
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.hook_handles = []
        
    def _hook_fn(self, module, input, output):
        """钩子函数，保存特征图"""
        self.feature_maps.append(output.detach())
    
    def register_hooks(self, layer_names=None):
        """
        注册钩子到指定层
        
        Args:
            layer_names: 要可视化的层名称列表，如果为None则可视化所有卷积层
        """
        if layer_names is None:
            # 自动找到所有卷积层
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    handle = module.register_forward_hook(self._hook_fn)
                    self.hook_handles.append(handle)
        else:
            for name in layer_names:
                layer = dict(self.model.named_modules())[name]
                handle = layer.register_forward_hook(self._hook_fn)
                self.hook_handles.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def visualize(self, input_image, save_path=None):
        """
        可视化特征图
        
        Args:
            input_image: 输入图像
            save_path: 保存路径
        """
        self.feature_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_image)
        
        # 可视化前几层的特征图
        num_layers = min(4, len(self.feature_maps))
        
        fig, axes = plt.subplots(num_layers, 8, figsize=(16, 2*num_layers))
        
        for layer_idx in range(num_layers):
            feature_map = self.feature_maps[layer_idx][0]  # 取第一个样本
            num_channels = min(8, feature_map.shape[0])
            
            for ch_idx in range(num_channels):
                ax = axes[layer_idx, ch_idx] if num_layers > 1 else axes[ch_idx]
                fm = feature_map[ch_idx].cpu().numpy()
                ax.imshow(fm, cmap='viridis')
                ax.axis('off')
                if ch_idx == 0:
                    ax.set_ylabel(f'Layer {layer_idx+1}', rotation=0, labelpad=40)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature maps saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_gradcam(model, images, labels, class_names, target_layer, 
                      save_dir='results', num_samples=9):
    """
    批量生成Grad-CAM可视化
    
    Args:
        model: 模型
        images: 图像张量，shape为(B, C, H, W)
        labels: 真实标签
        class_names: 类别名称列表
        target_layer: 目标层
        save_dir: 保存目录
        num_samples: 可视化的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 选择样本
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # 创建图形
    rows = int(np.ceil(num_samples / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, sample_idx in enumerate(indices):
        image = images[sample_idx:sample_idx+1]
        true_label = labels[sample_idx].item()
        
        # 生成CAM
        cam, predicted_class = grad_cam.generate_cam(image)
        
        # 准备显示
        img = image[0].cpu().numpy().transpose(1, 2, 0)
        
        # 反归一化（假设使用CIFAR-10的标准化）
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 调整CAM大小
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 叠加热力图
        superimposed = heatmap * 0.4 + img * 0.6
        superimposed = np.clip(superimposed, 0, 1)
        
        # 显示
        ax = axes[idx]
        ax.imshow(superimposed)
        ax.axis('off')
        
        # 标题
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[predicted_class]}'
        color = 'green' if true_label == predicted_class else 'red'
        ax.set_title(title, fontsize=10, color=color)
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to {save_path}")
    plt.close()
    
    # 清理
    grad_cam.remove_hooks()


def get_target_layer(model, model_name):
    """
    根据模型类型获取目标层
    
    Args:
        model: 模型
        model_name: 模型名称
        
    Returns:
        target_layer: 目标卷积层
    """
    model_name = model_name.lower()
    
    if 'resnet' in model_name:
        # ResNet系列：使用最后一个卷积层
        return model.layer4[-1].conv2
    elif 'vgg' in model_name:
        # VGG系列：使用features的最后一个卷积层
        for layer in reversed(list(model.features)):
            if isinstance(layer, nn.Conv2d):
                return layer
    elif 'custom' in model_name:
        # 自定义CNN：使用conv6
        return model.conv6
    elif 'mobilenet' in model_name:
        # MobileNet：使用最后一个卷积层
        return model.features[-1][0]
    else:
        # 默认：尝试找到最后一个卷积层
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("\nUsage:")
    print("  from grad_cam import GradCAM, visualize_gradcam, get_target_layer")
    print("  target_layer = get_target_layer(model, 'resnet18')")
    print("  visualize_gradcam(model, images, labels, class_names, target_layer)")
