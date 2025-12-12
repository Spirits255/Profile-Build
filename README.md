# Profile-Build
# 社交网络用户画像与推荐系统

基于深度学习的社交网络用户画像技术研究，集成了多种推荐算法模型，包括基线模型、深度神经网络模型和Transformer模型，用于电影评分预测和个性化推荐。

## 📋 项目概述

本项目是一个完整的推荐系统框架，支持以下功能：

- **多模型对比**：包括基线模型、深度神经网络模型和Transformer模型
- **完整的训练流程**：数据预处理、模型训练、验证和测试
- **详细评估**：多种评估指标和可视化分析
- **个性化推荐**：基于训练好的模型为用户生成电影推荐
- **实验管理**：自动创建实验目录，保存配置和结果

## 🏗️ 项目结构

```
├── config.py              # 配置文件管理
├── main.py                # 主程序入口
├── data_processor.py      # 数据加载与预处理
├── models.py              # 模型定义（包括Transformer）
├── trainer.py             # 模型训练与优化
├── evaluation.py          # 模型评估与比较
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd user-profiling-recommender

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

1. 下载 MovieLens-1M 数据集
2. 将数据集解压到项目根目录下的 `ml-1m` 文件夹
3. 确保包含以下文件：
   - `ratings.dat`
   - `users.dat`
   - `movies.dat`

### 3. 运行项目

#### 训练模式
```bash
python main.py
```
默认进入训练模式，将训练配置文件中指定的所有模型。

#### 评估模式
修改 `config.py` 中的 `MODE = 'evaluate'`，然后运行：
```bash
python main.py
```
这将评估已训练的模型并生成比较报告。

## ⚙️ 配置说明

### 主要配置项（config.py）

```python
# 实验模式
MODE = 'train'  # 'train' 或 'evaluate'

# 数据路径
DATA_PATH = "./ml-1m"

# 训练配置
TRAINING = {
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'use_amp': True,  # 混合精度训练
}

# 模型配置
MODELS = {
    'deep': {...},      # 深度神经网络
    'baseline': {...},  # 基线模型
    'transformer': {...} # Transformer模型
}
```

### Transformer 模型配置

```python
'transformer': {
    'embedding_dim': 64,      # 嵌入维度
    'nhead': 4,               # 多头注意力头数
    'num_layers': 2,          # 编码器层数
    'dim_feedforward': 256,   # 前馈网络维度
    'dropout': 0.1,           # Dropout率
}
```

## 🧠 模型架构

### 1. 基线模型 (BaselineModel)
- 简单的矩阵分解方法
- 用户和电影嵌入的交互学习

### 2. 深度神经网络模型 (UserProfilingModel)
- 包含用户特征和电影特征编码
- 多层全连接网络进行特征融合
- 支持用户画像学习

### 3. Transformer 模型 (TransformerRecommender)
- 基于 Transformer 编码器的推荐模型
- 将用户和电影特征作为 token 输入
- 多头注意力机制学习特征交互
- 适合捕捉复杂的用户-物品关系

## 📊 评估指标

项目提供多种评估指标：

1. **回归指标**
   - RMSE（均方根误差）
   - MAE（平均绝对误差）
   - R² 分数

2. **分类指标**
   - 准确率
   - 精确率、召回率、F1分数

3. **可视化分析**
   - 训练/验证损失曲线
   - 预测 vs 实际散点图
   - 误差分布直方图
   - 模型对比雷达图

## 📁 输出文件

训练完成后，会在 `experiment_results` 目录下生成：

```
experiment_results/
└── exp_YYYYMMDD_HHMMSS/
    ├── config.json              # 实验配置
    ├── models/                  # 保存的模型权重
    │   ├── baseline_model.pth
    │   ├── deep_model.pth
    │   └── transformer_model.pth
    ├── plots/                   # 可视化图表
    │   ├── training_loss_*.png
    │   ├── model_comparison.png
    │   └── training_history_*.png
    ├── reports/                 # 评估报告
    │   ├── data_statistics.json
    │   ├── *_metrics.json
    │   └── model_comparison_report.txt
    └── predictions/             # 预测结果
        ├── *_predictions.csv
        └── *_evaluation.csv
```

## 🔧 高级功能

### 1. 混合精度训练
- 使用 PyTorch AMP 加速训练
- 减少显存使用，加快训练速度

### 2. 动态学习率调整
- ReduceLROnPlateau 调度器
- 根据验证损失自动调整学习率

### 3. 早停机制
- 监控验证损失
- 当性能不再提升时自动停止训练

### 4. 梯度裁剪
- 防止梯度爆炸
- 提高训练稳定性

## 💡 使用示例

### 自定义训练配置

```python
# 修改 config.py 中的配置
TRAINING = {
    'epochs': 100,
    'batch_size': 256,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
}

# 选择要训练的模型
EVALUATION = {
    'models_to_evaluate': ['transformer', 'deep'],  # 只训练这两个模型
}
```

### 使用特定实验进行评估

```python
# 修改 config.py
MODE = 'evaluate'
EXPERIMENT_NAME = 'exp_20231215_143000'  # 指定实验名称
```

## 🛠️ 开发指南

### 添加新模型

1. 在 `models.py` 中定义新模型类
2. 继承 `nn.Module` 基类
3. 在 `__init__` 方法中定义网络结构
4. 实现 `forward` 方法
5. 在 `config.py` 的 `MODELS` 部分添加配置
6. 在 `main.py` 的训练和评估逻辑中添加模型支持

### 修改数据处理

1. 在 `data_processor.py` 中修改特征工程方法
2. 添加新的特征提取逻辑
3. 调整数据过滤条件

## 📈 性能优化建议

1. **硬件要求**
   - GPU：推荐 NVIDIA GPU（支持 CUDA）
   - 内存：至少 8GB RAM
   - 存储：至少 2GB 可用空间

2. **训练优化**
   - 启用混合精度训练（默认开启）
   - 调整批次大小以适应显存
   - 使用更大的嵌入维度以提升性能

3. **数据处理**
   - 调整数据过滤阈值
   - 添加更多特征工程
   - 考虑使用更大的数据集

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小
   - 降低嵌入维度
   - 使用更小的数据集

2. **训练不收敛**
   - 调整学习率
   - 检查数据预处理
   - 验证模型架构

3. **评估模式找不到模型**
   - 确保已运行训练模式生成模型
   - 检查实验路径是否正确

### 调试建议

- 查看控制台输出的错误信息
- 检查数据路径和文件格式
- 验证配置参数的有效性

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

1. Fork 项目仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📚 参考文献

1. MovieLens 1M Dataset
2. Transformer: Attention Is All You Need
3. Neural Collaborative Filtering
4. Deep Learning for Recommender Systems

---

**提示**：首次运行前，请确保已正确配置数据路径和依赖环境。建议先使用较小的配置进行测试，确认无误后再进行完整训练。
