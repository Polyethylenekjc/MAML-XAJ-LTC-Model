import yaml
import pandas as pd

# ======== 参数设置 ========
yaml_path = "/home/fifth/code/Python/GTLF/Config/test.yaml"   # ✅ 替换为你的 YAML 文件路径
csv_path = "/home/fifth/code/Python/GTLF/res/analysis/missing_predictions_report.csv"     # ✅ 替换为你的 CSV 文件路径
save_path = "/home/fifth/code/Python/GTLF/res/analysis/config_updated.yaml"  # ✅ 输出保存路径

# ======== 读取数据 ========
with open(yaml_path, "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)

df = pd.read_csv(csv_path)
scenes_to_enable = set(df["scene"].astype(str))

data = yaml_data.get("data").get("datasets")

# ======== 修改 enabled 字段 ========
for dataset in data:
    name = str(dataset.get("name", ""))
    if name in scenes_to_enable:
        dataset["enabled"] = True

# ======== 保存修改后的 YAML ========
with open(save_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

print(f"✅ 已更新并保存到: {save_path}")
