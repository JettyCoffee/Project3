"""测试配置序列化"""
import torch
from config import Config

# 测试get_config_dict是否可以序列化
config_dict = Config.get_config_dict()

print("配置字典内容：")
for k, v in list(config_dict.items())[:10]:
    print(f"  {k}: {v} ({type(v).__name__})")

# 测试pickle
import pickle
try:
    serialized = pickle.dumps(config_dict)
    deserialized = pickle.loads(serialized)
    print("\n✅ 配置字典可以成功序列化！")
except Exception as e:
    print(f"\n❌ 序列化失败: {e}")

# 测试torch.save
try:
    torch.save({'config': config_dict}, '/tmp/test_config.pth')
    loaded = torch.load('/tmp/test_config.pth')
    print("✅ torch.save 成功！")
except Exception as e:
    print(f"❌ torch.save 失败: {e}")
