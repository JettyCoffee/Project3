#!/bin/bash
# 快速启动脚本 - CIFAR-10 图像分类项目

echo "=========================================="
echo "CIFAR-10 Image Classification Project"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# 检查是否已安装依赖
if ! python -c "import torch" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "✓ Dependencies installed"
echo ""

# 显示菜单
echo "Please select an option:"
echo "1) Quick Start (Train ResNet-18 for 20 epochs)"
echo "2) Full Training (Train ResNet-18 for 100 epochs)"
echo "3) Train Custom CNN"
echo "4) Evaluate existing model"
echo "5) Generate visualizations only"
echo "6) Compare multiple models"
echo "7) Exit"
echo ""

read -p "Enter your choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting quick training (20 epochs)..."
        python main.py --mode full --model resnet18 --epochs 20
        ;;
    2)
        echo ""
        echo "🚀 Starting full training (100 epochs)..."
        python main.py --mode full --model resnet18 --epochs 100
        ;;
    3)
        echo ""
        echo "🚀 Training Custom CNN..."
        python main.py --mode full --model custom_cnn --epochs 80
        ;;
    4)
        echo ""
        echo "📊 Evaluating model..."
        python main.py --mode eval --visualize
        ;;
    5)
        echo ""
        echo "📈 Generating visualizations..."
        python main.py --mode visualize
        ;;
    6)
        echo ""
        echo "🔄 Comparing models (this will take time)..."
        echo ""
        echo "Training Custom CNN..."
        python main.py --mode train --model custom_cnn --epochs 50
        mv checkpoints/best_model.pth checkpoints/custom_cnn_best.pth
        
        echo ""
        echo "Training ResNet-18..."
        python main.py --mode train --model resnet18 --epochs 50
        mv checkpoints/best_model.pth checkpoints/resnet18_best.pth
        
        echo ""
        echo "Training VGG-16..."
        python main.py --mode train --model vgg16 --epochs 50
        mv checkpoints/best_model.pth checkpoints/vgg16_best.pth
        
        echo ""
        echo "✓ All models trained. Check checkpoints/ directory for results."
        ;;
    7)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Done!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - checkpoints/  (trained models)"
echo "  - logs/         (training history)"
echo "  - results/      (evaluation results and visualizations)"
echo ""
echo "Check README.md for more usage instructions."
