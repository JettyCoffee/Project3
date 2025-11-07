"""
Grad-CAM可视化演示脚本
用于展示模型关注的区域
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from models import get_model
from data_loader import get_data_loaders
from grad_cam import GradCAM, visualize_gradcam, get_target_layer, FeatureMapVisualizer


def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('config', {})
    model_name = model_config.get('MODEL_NAME', 'resnet18')
    num_classes = model_config.get('NUM_CLASSES', 10)
    
    # 创建模型
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model_name}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    return model, model_name


def visualize_single_gradcam(model, model_name, image, true_label, class_names, 
                            target_layer, save_path=None):
    """为单个图像生成Grad-CAM可视化"""
    import cv2
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 生成CAM
    with torch.no_grad():
        output = model(image)
    predicted_class = output.argmax(dim=1).item()
    
    cam, _ = grad_cam.generate_cam(image)
    
    # 准备显示
    img = image[0].cpu().numpy().transpose(1, 2, 0)
    
    # 反归一化
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
    
    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 热力图
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # 叠加图
    axes[2].imshow(superimposed)
    axes[2].set_title('Superimposed', fontsize=12)
    axes[2].axis('off')
    
    # 颜色条说明
    axes[3].imshow(heatmap)
    axes[3].set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_class]}', 
                     fontsize=12, color='green' if true_label == predicted_class else 'red')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    grad_cam.remove_hooks()


def visualize_feature_maps(model, model_name, image, save_path=None):
    """可视化特征图"""
    print("Generating feature map visualizations...")
    
    visualizer = FeatureMapVisualizer(model)
    visualizer.register_hooks()
    visualizer.visualize(image, save_path)
    visualizer.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for CIFAR-10')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--num-samples', type=int, default=9,
                       help='可视化的样本数量')
    parser.add_argument('--save-dir', type=str, default='results/gradcam',
                       help='保存目录')
    parser.add_argument('--feature-maps', action='store_true',
                       help='是否生成特征图可视化')
    parser.add_argument('--single-image', type=int, default=None,
                       help='可视化单张图像的索引')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # 加载模型
    model, model_name = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载数据
    print("\nLoading data...")
    _, _, test_loader = get_data_loaders()
    
    # 获取一批数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    # 获取目标层
    print(f"\nGetting target layer for {model_name}...")
    target_layer = get_target_layer(model, model_name)
    
    if target_layer is None:
        print("Error: Could not find target layer for Grad-CAM")
        return
    
    print(f"Target layer: {target_layer}")
    
    # 生成Grad-CAM可视化
    if args.single_image is not None:
        # 单张图像详细可视化
        idx = args.single_image
        single_image = images[idx:idx+1]
        true_label = labels[idx].item()
        
        save_path = os.path.join(args.save_dir, f'single_gradcam_{idx}.png')
        visualize_single_gradcam(model, model_name, single_image, true_label, 
                                Config.CLASS_NAMES, target_layer, save_path)
        
        # 特征图可视化
        if args.feature_maps:
            feature_save_path = os.path.join(args.save_dir, f'feature_maps_{idx}.png')
            visualize_feature_maps(model, model_name, single_image, feature_save_path)
    else:
        # 批量可视化
        print(f"\nGenerating Grad-CAM for {args.num_samples} samples...")
        visualize_gradcam(model, images, labels, Config.CLASS_NAMES, target_layer,
                         save_dir=args.save_dir, num_samples=args.num_samples)
    
    print("\n" + "="*60)
    print("Grad-CAM visualization completed!")
    print(f"Results saved in: {args.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
