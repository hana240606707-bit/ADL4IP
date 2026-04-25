# Faster R-CNN 海洋生物检测

## 项目概述

这是一个使用 **Faster R-CNN** 进行海洋生物检测的完整项目。项目针对严重不平衡的数据集进行了优化处理。

**数据集信息:**
- **类别**: fish, jellyfish, penguin, shark, puffin, stingray, starfish (7个)
- **格式**: COCO JSON标注
- **不平衡比率**: 23.02x (Fish 2,669 vs Starfish 116)
- **模型**: Faster R-CNN + ResNet50-FPN
- **评估指标**: mAP (COCO format)

## 项目结构

```
ADL4IP/
├── notebooks/
│   └── faster_rcnn_training.ipynb    # Google Colab完整训练脚本
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # COCO数据加载 + 增强
│   ├── model.py                      # Faster R-CNN模型
│   ├── train.py                      # 训练管道 + 损失加权
│   ├── evaluate.py                   # mAP评估
│   └── utils.py                      # 工具函数
├── configs/
│   └── config.yaml                   # 训练配置
├── requirements.txt                  # 依赖包
└── README.md                         # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Google Colab 快速开始

在Google Colab中打开 `notebooks/faster_rcnn_training.ipynb`，按顺序运行所有单元格：

```
1. 安装依赖
2. 挂载Google Drive
3. 克隆项目代码
4. 导入库
5. 数据分析
6. 类别权重计算
7. 创建数据加载器
8. 创建模型
9. 训练配置
10. 开始训练（主要步骤）
11. 加载最佳模型
12. 模型评估
13. 结果总结
```

### 3. 准备数据

确保你的数据集结构如下：

```
dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/
│   │   └── ...
│   └── annotations.json
└── test/
    ├── images/
    │   └── ...
    └── annotations.json
```

## 不平衡数据处理

该项目实现了3层策略来处理类别不平衡：

### 1. 加权损失函数 ✅

```python
# 在训练中增加分类损失权重
loss_dict[k] = v * 1.5  # 分类损失权重 = 1.5x
```

### 2. 类别权重

使用反向频率加权：
```
weight = total / (num_classes * count)
```

例如，对于该数据集：
- **Starfish** (116): 23.02x 权重
- **Stingray** (184): 14.51x 权重
- **Fish** (2,669): 1.0x 权重

### 3. 数据增强

针对训练集应用：
- 水平翻转 (p=0.5)
- 垂直翻转 (p=0.3)
- 随机旋转 ±15° (p=0.5)
- 亮度/对比度调整 (p=0.3)
- 颜色抖动 (p=0.3)
- 高斯噪声 (p=0.2)

## 模型架构

**Faster R-CNN** 组件：

1. **主干网络**: ResNet50-FPN
   - 提取多尺度特征
   - 预训练权重：ImageNet

2. **区域提案网络 (RPN)**
   - 生成候选边界框
   - 边界框回归

3. **ROI头部**
   - 分类器（7类 + 背景）
   - 边界框回归器

## 训练配置

默认配置：

| 参数 | 值 |
|------|-----|
| 批大小 | 4 |
| 学习率 | 0.005 |
| 优化器 | SGD (momentum=0.9) |
| 权重衰减 | 0.0005 |
| 学习率调度 | StepLR (step_size=10, gamma=0.1) |
| Epochs | 30 |
| 数据增强 | 随机翻转、旋转、颜色抖动 |

可在 `configs/config.yaml` 中修改。

## 预期结果

基于类别分布的预期性能：

| 类别 | 样本数 | 预期mAP |
|------|-------|---------|
| Fish | 2,669 | 0.85-0.95 |
| Jellyfish | 694 | 0.65-0.75 |
| Penguin | 516 | 0.60-0.70 |
| Shark | 354 | 0.50-0.65 |
| Puffin | 284 | 0.45-0.60 |
| Stingray | 184 | 0.40-0.55 |
| Starfish | 116 | 0.30-0.50 |
| **总体** | **4,817** | **0.60-0.75** |

## 评估指标

使用 **COCO mAP** 标准：

- **mAP@0.5:0.95**: IoU阈值从0.5到0.95的平均
- **mAP@0.5**: 严格IoU=0.5的mAP
- **mAP@0.75**: 严格IoU=0.75的mAP
- **mAP (small/medium/large)**: 不同大小目标的mAP

## 文件说明

### `src/data_loader.py`
- `COCODataset`: 自定义COCO数据集类
- `get_train_transforms()`: 训练增强管道
- `get_val_transforms()`: 验证增强管道
- `create_data_loaders()`: 创建DataLoader

### `src/model.py`
- `create_model()`: 创建Faster R-CNN模型
- `get_model_info()`: 获取模型统计信息

### `src/train.py`
- `Trainer`: 完整的训练管道
- 特性：
  - 损失加权处理不平衡
  - 学习率调度
  - 检查点保存
  - 验证循环

### `src/evaluate.py`
- `COCOEvaluator`: COCO格式评估
- mAP计算和逐类统计

### `src/utils.py`
- `compute_class_weights()`: 计算类别权重
- `plot_class_distribution()`: 可视化类别分布
- `get_logger()`: 日志记录

## 常见问题

### Q: 训练太慢？
**A:** 
- 减小批大小（但会影响性能）
- 减少训练epochs
- 使用更少的数据增强操作
- 检查GPU占用

### Q: mAP太低？
**A:**
- 增加训练epochs
- 调整学习率
- 检查数据标注质量
- 增加数据增强
- 使用更大的模型

### Q: 显存不足？
**A:**
```python
# 在config.yaml中修改
batch_size: 2  # 从4改为2
trainable_backbone_layers: 1  # 冻结更多层
```

### Q: 少数类性能太差？
**A:**
- 增加类别权重（在utils.py中）
- 对少数类应用额外的数据增强
- 使用焦点损失（focal loss）

## 参考文献

- [Faster R-CNN 论文](https://arxiv.org/abs/1506.01497)
- [PyTorch官方实现](https://pytorch.org/vision/stable/models.html#detection)
- [COCO数据集](https://cocodataset.org/)
- [Detectron2 文档](https://detectron2.readthedocs.io/)

## 许可证

MIT License

## 作者

这个项目是ADL4IP课程的一部分。
