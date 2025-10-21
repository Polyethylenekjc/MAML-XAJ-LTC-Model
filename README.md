# GTLF 项目文档

## 项目概述
GTLF（General Time Series Learning Framework）是一个通用的时间序列学习框架，专注于洪水预测和其他时间序列分析任务。该项目结合了深度学习模型和物理元学习方法，以提高在极端事件（如洪水期）的预测精度。

## 主要功能
- 支持多种时间序列模型，包括 GRKU、LSTM 和 GRU。
- 提供基于 SHAP 的特征解释功能。
- 实现了元学习策略以提升模型泛化能力。
- 包含多种数据预处理和特征工程工具。

## 技术栈
- Python 3.x
- PyTorch
- SHAP
- FAISS
- NumPy, Pandas, Scikit-learn

## 项目结构
```
├── Config/              # 配置文件目录
├── src/                 # 源代码目录
│   ├── analyzer/        # 分析器模块
│   ├── data/            # 数据处理模块
│   ├── model/           # 模型定义
│   ├── trainer/         # 训练逻辑
│   └── utils/           # 工具类
├── README.md            # 项目说明文档
└── config_schema.json   # 配置文件模式
```

## 配置指南

### 配置文件结构说明
- `data`: 数据集配置部分
  - `default_dataset`: 默认数据集配置
  - `datasets`: 具体数据集列表
- `models`: 模型配置部分，包含各种模型的具体配置
- `analyzers`: 分析器配置列表
- `output`: 输出路径配置

### 创建新组件指南

#### 创建新训练器
1. 在 `src/trainer/` 目录下创建新的训练器类文件，继承自 BaseModelTrainer
2. 实现 `train` 和 `predict` 方法
3. 在配置文件中添加新的训练器配置，指定 `trainer.type` 为新训练器类名

#### 创建新模型
1. 在 `src/model/` 目录下创建新的模型类文件
2. 继承适当的基类（如 nn.Module）
3. 实现模型结构和前向传播逻辑
4. 在 ModelFactory 中注册新模型类型
5. 在配置文件中添加新的模型配置，设置 `type` 字段为新模型类名

#### 创建新分析器
1. 在 `src/analyzer/` 目录下创建新的分析器类文件，继承自 AnalyzerBase
2. 实现 `analyze` 方法和分析逻辑
3. 在 AnalyzerFactory 中注册新分析器类型
4. 在配置文件的 `analyzers` 列表中添加新分析器配置

#### 创建新数据处理器
1. 在 `src/data/` 目录下创建新的数据处理器类文件，继承自 DataProcessorBase
2. 实现数据处理逻辑
3. 在 DataProcessorFactory 中注册新处理器类型
4. 在数据集配置的 `processors` 列表中添加新处理器配置