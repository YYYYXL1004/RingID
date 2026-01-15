# RingID 频段解耦实验

> 基于 [RingID](http://arxiv.org/abs/2404.14055) 的多供应商识别场景下频段策略验证实验

## 项目背景

本项目是**成员 B** 的实验代码，用于验证成员 A 在 Tree-Ring 上发现的**双频带策略**在 RingID 多供应商识别场景下是否同样有效。

| 阶段 | 负责人 | 研究内容 | 场景 |
|------|--------|----------|------|
| 假设提出 | 成员 A | 双频带策略提升鲁棒性 | Tree-Ring |
| **假设验证** | **成员 B** | 双频带策略是否同样有效？ | **RingID** |

## 核心实验结果

| 策略 | Clean | C&S75 | 平均准确率 |
|------|-------|-------|------------|
| 低频 (R=3-7) | 100% | 88% | 97.7% |
| 高频 (R=8-14) | 100% | 80% | 96.6% |
| **双频带 (R=3-14)** | 100% | **90%** | **98.3%** ✓ |

**结论**：双频带策略在多供应商识别场景下**同样表现最优**，验证了其普适性。

## 环境配置

```bash
conda activate tr
```

**依赖版本**：
- PyTorch >= 1.13.0
- diffusers == 0.11.1
- transformers == 4.23.1

## 使用方法

### 多密钥识别实验
```bash
python identify.py --run_name test --trials 50 --assigned_keys 5 --gpu_id 3
```

### 频段选择对比实验
```bash
python scripts/frequency_band_test.py --gpu 3
```

### 密钥容量测试
```bash
python scripts/key_capacity_test.py --keys 5,10,20,30,50 --gpu 3
```

## 项目结构

```
RingID/
├── identify.py                 # 主程序：多密钥识别
├── utils.py                    # 核心工具函数
├── inverse_stable_diffusion.py # DDIM 逆向
├── modified_stable_diffusion.py# 修改版 SD pipeline
├── scripts/
│   ├── frequency_band_test.py  # 频段对比实验
│   ├── key_capacity_test.py    # 密钥容量测试
│   └── generate_confusion_matrix.py
├── docs/
│   ├── Identifiability_CN.md   # 中文报告
│   └── Identifiability_EN.md   # 英文报告
├── runs/                       # 实验结果
├── models/                     # SD 模型
└── sd_prompts_dataset/         # Prompt 数据集
```

## 文档

- [中文报告](docs/Identifiability_CN.md)
- [English Report](docs/Identifiability_EN.md)

## 参考

- [RingID Paper](http://arxiv.org/abs/2404.14055)
- [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark)


