import os
import shutil
from datetime import datetime

# 获取当前时间戳
timestamp = datetime.now().strftime("%m%d_%H%M")

# 原始文件路径
original_checkpoint_path = "./checkpoints/best_model.pth"

# 新文件路径，加上时间戳
new_checkpoint_path = f"./checkpoints/best_model_{timestamp}.pth"

# 重命名文件
os.rename(original_checkpoint_path, new_checkpoint_path)

print(f"文件已重命名: {original_checkpoint_path} -> {new_checkpoint_path}")

# 创建新文件夹
new_folder = f"./results/{timestamp}"
os.makedirs(new_folder, exist_ok=True)

# 复制 results/ 下的所有文件到新文件夹
for file in os.listdir("./results"):
    if os.path.isfile(os.path.join("./results", file)):
        shutil.copy(os.path.join("./results", file), os.path.join(new_folder, file))

print(f"Results files copied to {new_folder}")