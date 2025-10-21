import os

# ======== 参数设置 ========
folder_path = "/home/fifth/code/Python/GTLF/data/meta"  # ✅ 替换为你的文件夹路径
processor_anchor = "*id001"
target_column = "runoffs"
test_size = 0.2
window = 15

# ======== 遍历文件并生成配置 ========
configs = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        name_part = file.split("_")[-1]  # 获取最后一个 "_" 之后的部分
        name = name_part.replace(".csv", "")  # 去掉扩展名
        config = f"""- enabled: true
  name: {name}
  path: {os.path.join(folder_path, file)}
  processors: {processor_anchor}
  target_column: {target_column}
  test_size: {test_size}
  window: {window}
"""
        configs.append(config)

# ======== 输出到控制台 ========
print("\n".join(configs))
