"""
Grad-CAM可视化模块 - 展示模型关注的区域
包括：Grad-CAM、特征图可视化等
完全重写版本，使用更可靠的实现方式
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
    使用简单的hook和retain_grad方法，避免复杂的backward hook问题
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: 要可视化的模型
            target_layer: 目标卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = None
        
        # 注册前向hook
        self._register_hook()
        
    def _register_hook(self):
        """只注册前向hook来捕获激活值"""
        def forward_hook(module, input, output):
            # 直接保存激活值的引用（不detach）
            # 调用retain_grad以保留非叶子节点的梯度
            self.activations = output
            if output.requires_grad:
                output.retain_grad()
        
        self.hook = self.target_layer.register_forward_hook(forward_hook)
    
    def remove_hooks(self):
        """移除hook"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def generate_cam(self, input_image, target_class=None):
        """
        生成类激活图
        
        Args:
            input_image: 输入图像，shape为(1, C, H, W)
            target_class: 目标类别，如果为None则使用预测类别
            
        Returns:
            cam: 类激活图 (numpy array)
            predicted_class: 预测类别
        """
        # 清空之前的激活值
        self.activations = None
        
        # 获取设备
        device = next(self.model.parameters()).device
        
        # 准备输入
        input_tensor = input_image.clone().detach().to(device).float()
        input_tensor.requires_grad = True
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 临时禁用Cutout
        cutout_backup = None
        if hasattr(self.model, 'cutout'):
            cutout_backup = self.model.cutout
            self.model.cutout = None
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 获取预测类别
        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class
        
        # 检查激活值是否被捕获
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check if hook is properly registered.")
        
        # 清零梯度
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # 计算目标类别的得分并反向传播
        target_score = output[0, target_class]
        target_score.backward(retain_graph=False)
        
        # 恢复Cutout
        if cutout_backup is not None:
            self.model.cutout = cutout_backup
        
        # 获取激活值的梯度
        gradients = self.activations.grad
        
        if gradients is None:
            raise RuntimeError("Gradients not computed. Check model architecture.")
        
        # 移到CPU并detach进行后续计算
        gradients = gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # 计算权重：对梯度在空间维度上进行全局平均池化
        # gradients shape: [1, C, H, W]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和得到CAM
        # activations shape: [1, C, H, W]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        
        # 应用ReLU（只保留正值）
        cam = F.relu(cam)
        
        # 转换为numpy并归一化
        cam = cam.squeeze().numpy()
        
        # 避免除零错误
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam, predicted_class
    
    def __del__(self):
        """析构函数"""
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
                      save_dir='results/gradcam', num_samples=9):
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
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 选择样本
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # 创建图形
    rows = int(np.ceil(num_samples / 3))
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    # 确保axes是二维数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    print(f"Generating {num_samples} Grad-CAM visualizations...")
    
    for idx, sample_idx in enumerate(indices):
        try:
            # 获取单个图像
            image = images[sample_idx:sample_idx+1]
            true_label = labels[sample_idx].item()
            
            # 生成CAM
            cam, predicted_class = grad_cam.generate_cam(image)
            
            # 准备原始图像显示
            img = image[0].cpu().numpy().transpose(1, 2, 0)
            
            # 反归一化（CIFAR-10标准化参数）
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # 将CAM调整到与图像相同的大小
            cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
            
            # 创建热力图（使用JET colormap）
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # 叠加热力图和原始图像
            superimposed = heatmap * 0.4 + img * 0.6
            superimposed = np.clip(superimposed, 0, 1)
            
            # 显示
            ax = axes[idx]
            ax.imshow(superimposed)
            ax.axis('off')
            
            # 设置标题（正确预测用绿色，错误用红色）
            title = f'True: {class_names[true_label]}\nPred: {class_names[predicted_class]}'
            color = 'green' if true_label == predicted_class else 'red'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            
            print(f"  ✓ Sample {idx+1}/{num_samples} completed")
            
        except Exception as e:
            print(f"  ✗ Sample {idx+1}/{num_samples} failed: {str(e)}")
            ax = axes[idx]
            ax.axis('off')
            ax.text(0.5, 0.5, f'Failed\n{str(e)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=8, color='red')
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grad-CAM visualization saved to {save_path}")
    plt.close()
    
    # 清理
    grad_cam.remove_hooks()


def visualize_gradcam_per_class(model, data_loader, class_names, target_layer, 
                                save_dir='results/gradcam', samples_per_class=1):
    """
    为每个类别生成Grad-CAM可视化，确保包含所有10个类别
    
    Args:
        model: 模型
        data_loader: 数据加载器
        class_names: 类别名称列表
        target_layer: 目标层
        save_dir: 保存目录
        samples_per_class: 每个类别的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    num_classes = len(class_names)
    
    # 收集每个类别的样本
    class_samples = {i: [] for i in range(num_classes)}
    
    print(f"Collecting samples for each class...")
    device = next(model.parameters()).device
    
    # 遍历数据集收集每个类别的样本
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(len(images)):
            label = labels[i].item()
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append((images[i:i+1], label))
        
        # 检查是否所有类别都收集到足够的样本
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    # 检查每个类别是否都有样本
    missing_classes = [i for i in range(num_classes) if len(class_samples[i]) == 0]
    if missing_classes:
        print(f"⚠ Warning: No samples found for classes: {[class_names[i] for i in missing_classes]}")
    
    # 创建可视化网格
    rows = num_classes
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    # 确保axes是二维数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"\nGenerating Grad-CAM for all {num_classes} classes...")
    
    # 为每个类别生成可视化
    for class_idx in range(num_classes):
        samples = class_samples[class_idx]
        
        for sample_idx in range(samples_per_class):
            # 正确处理二维索引
            if isinstance(axes, np.ndarray) and axes.ndim == 2:
                ax = axes[class_idx, sample_idx]
            elif isinstance(axes, np.ndarray) and axes.ndim == 1:
                ax = axes[class_idx]
            else:
                ax = axes
            
            if sample_idx < len(samples):
                try:
                    image, true_label = samples[sample_idx]
                    
                    # 生成CAM
                    cam, predicted_class = grad_cam.generate_cam(image)
                    
                    # 准备原始图像显示
                    img = image[0].cpu().numpy().transpose(1, 2, 0)
                    
                    # 反归一化（CIFAR-10标准化参数）
                    mean = np.array([0.4914, 0.4822, 0.4465])
                    std = np.array([0.2470, 0.2435, 0.2616])
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    
                    # 将CAM调整到与图像相同的大小
                    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
                    
                    # 创建热力图（使用JET colormap）
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # 叠加热力图和原始图像
                    superimposed = heatmap * 0.4 + img * 0.6
                    superimposed = np.clip(superimposed, 0, 1)
                    
                    # 显示
                    ax.imshow(superimposed)
                    ax.axis('off')
                    
                    # 设置标题（正确预测用绿色，错误用红色）
                    title = f'{class_names[true_label]}\nPred: {class_names[predicted_class]}'
                    color = 'green' if true_label == predicted_class else 'red'
                    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
                    
                    print(f"  ✓ Class {class_idx+1}/{num_classes} ({class_names[class_idx]}), Sample {sample_idx+1}/{samples_per_class} completed")
                    
                except Exception as e:
                    print(f"  ✗ Class {class_idx+1}/{num_classes} ({class_names[class_idx]}), Sample {sample_idx+1}/{samples_per_class} failed: {str(e)}")
                    ax.axis('off')
                    ax.text(0.5, 0.5, f'Failed\n{str(e)[:50]}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8, color='red')
            else:
                # 没有足够的样本
                ax.axis('off')
                ax.text(0.5, 0.5, 'No sample', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_all_classes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grad-CAM visualization for all classes saved to {save_path}")
    plt.close()
    
    # 清理
    grad_cam.remove_hooks()
    
    return class_samples


def get_target_layer(model, model_name):
    """
    根据模型类型获取目标层（用于Grad-CAM）
    
    Args:
        model: 模型
        model_name: 模型名称
        
    Returns:
        target_layer: 目标卷积层（通常是最后一个卷积层）
    """
    model_name = model_name.lower()
    
    print(f"Finding target layer for model: {model_name}")
    
    try:
        if 'wide_resnet' in model_name:
            # Wide ResNet：使用layer3的最后一个块的第二个卷积层
            target = model.layer3[-1].conv2
            print(f"  → Using layer3[-1].conv2")
            return target
            
        elif 'resnet' in model_name:
            # ResNet系列：使用layer4的最后一个块
            if hasattr(model, 'layer4'):
                target = model.layer4[-1].conv2
                print(f"  → Using layer4[-1].conv2")
                return target
            elif hasattr(model, 'layer3'):
                target = model.layer3[-1].conv2
                print(f"  → Using layer3[-1].conv2")
                return target
            
    except Exception as e:
        print(f"  Error finding specific layer: {e}")
    
    # 默认方案：找到最后一个Conv2d层
    print(f"  → Using fallback: finding last Conv2d layer")
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name
    
    if last_conv is not None:
        print(f"  → Found: {last_name}")
        return last_conv
    else:
        raise ValueError(f"Could not find any Conv2d layer in model {model_name}")


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("\nUsage:")
    print("  from grad_cam import GradCAM, visualize_gradcam, get_target_layer")
    print("  target_layer = get_target_layer(model, 'resnet18')")
    print("  visualize_gradcam(model, images, labels, class_names, target_layer)")
