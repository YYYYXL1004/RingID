# 成员 B 详细执行计划

## 角色定位
**挑战者 (The Challenger)** —— 负责"异"

## 核心任务
攻克"供应商可识别性"，探究密钥安全性

---

## 一、环境配置 (Day 1)

### 1.1 激活 conda 环境
```bash
conda activate tr
```

### 1.2 验证依赖版本
```bash
python -c "import torch; import diffusers; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'diffusers: {diffusers.__version__}'); print(f'transformers: {transformers.__version__}')"
```

**要求版本**：
- PyTorch >= 1.13.0
- diffusers == 0.11.1
- transformers == 4.23.1

### 1.3 Stable Diffusion 模型（已配置）
模型已通过 ModelScope 下载到本地：
```
./models/AI-ModelScope/stable-diffusion-2-1-base/
```

代码已修改为自动使用本地模型，无需网络连接。

### 1.4 跑通 Demo
```bash
python identify.py --run_name test_demo --trials 5 --assigned_keys 3 --reference_model None
```

**注意**：不需要 `--online` 参数，代码会自动使用本地模型。

---

## 二、理解 Multi-Key 逻辑 (Day 1)

### 2.1 关键文件
| 文件 | 作用 |
|------|------|
| `identify.py` | 主程序，Multi-Key 识别实验 |
| `utils.py` | 核心函数，包含 `make_Fourier_ringid_pattern()` |
| `verify.py` | 单 Key 验证实验 |

### 2.2 关键参数 (在 `utils.py` 中)
```python
RADIUS = 14              # 外圈半径
RADIUS_CUTOFF = 3        # 内圈半径（截断）
RING_WATERMARK_CHANNEL = [3]   # 使用的通道
```

### 2.3 Multi-Key 生成原理
- 每个 Key 由 `(RADIUS - RADIUS_CUTOFF) = 11` 个环组成
- 每个环可以取不同的值（由 `--ring_value_range` 和 `--num_inmost_keys` 控制）
- 不同的值组合 = 不同的 Key

---

## 三、生成 Multi-Key 数据 (Day 2 - 核心任务)

### 3.1 目标
生成 **5 个不同 Key**，每个 Key 生成 **20 张图片**

### 3.2 执行命令
```bash
python identify.py \
    --run_name multi_key_5vendors \
    --trials 100 \
    --assigned_keys 5 \
    --num_images 1 \
    --save_generated_imgs 1
```

**注意**：不需要 `--online` 参数，代码已配置使用本地 ModelScope 模型。

**参数说明**：
- `--assigned_keys 5`：随机选择 5 个不同的 Key
- `--trials 100`：生成 100 张图（每个 Key 约 20 张）
- `--save_generated_imgs 1`：保存生成的图片

### 3.3 输出目录结构
```
runs/
└── YYYY_MM_DD_HH_MM_SS_multi_key_5vendors/
    ├── images/
    │   ├── watermarked/
    │   │   ├── Key_0.Prompt_0.Fourier_watermark.ClipSim_0.xxxx.jpg
    │   │   ├── Key_1.Prompt_1.Fourier_watermark.ClipSim_0.xxxx.jpg
    │   │   └── ...
    │   └── no_watermark/
    │       └── ...
    └── log.csv
```

---

## 四、正交性测试 (Day 2)

### 4.1 目标
测试 Key A 的检测器是否会误报 Key B 的图片

### 4.2 原理
`identify.py` 已经实现了这个逻辑：
1. 对每张图片，用所有 Key 的检测器计算距离
2. 选择距离最小的 Key 作为预测结果
3. 比较预测结果与真实 Key，计算准确率

### 4.3 结果解读
程序会输出类似：
```
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
|     Id Acc     | Clean | Rot 75 | JPEG 25 | C&S 75 | Blur 8 | Noise 0.1 | Brightness... |  Avg  |
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
| L1 |a-b|       | 0.950 | 0.820  | 0.780   | 0.850  | 0.900  |   0.880   |     0.750     | 0.847 |
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
```

- **Clean**：无攻击时的识别准确率
- **其他列**：各种攻击下的识别准确率
- **Avg**：平均准确率

---

## 五、生成混淆矩阵 (Day 2-3)

### 5.1 需要修改代码
在 `identify.py` 中添加混淆矩阵统计逻辑（或写一个新脚本）

### 5.2 混淆矩阵格式
```
          Predicted Key
          K0   K1   K2   K3   K4
Actual K0 [18,  1,   0,   1,   0]
       K1 [ 0, 19,   1,   0,   0]
       K2 [ 1,  0,  18,   1,   0]
       K3 [ 0,  0,   1,  19,   0]
       K4 [ 0,  1,   0,   0,  19]
```

### 5.3 可视化代码示例
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设 confusion_matrix 是你统计的结果
confusion_matrix = np.array([...])

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Key {i}' for i in range(5)],
            yticklabels=[f'Key {i}' for i in range(5)])
plt.xlabel('Predicted Key')
plt.ylabel('Actual Key')
plt.title('Multi-Key Identification Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
```

---

## 六、写作任务 (Day 4)

### 6.1 撰写 Identifiability 章节

**结构建议**：

1. **问题引入**：Tree-Ring 只能做二分类（有/无水印），无法区分不同供应商
2. **RingID 方案**：介绍 Multi-Key 机制
3. **实验设置**：5 个 Key，每个 20 张图，统一 Prompt
4. **结果分析**：
   - 展示混淆矩阵热力图
   - 分析对角线（正确识别率）
   - 分析非对角线（误报情况）
5. **结论**：RingID 在多供应商场景下的有效性

### 6.2 关键图表
- [ ] 混淆矩阵热力图
- [ ] 不同攻击下的识别准确率柱状图

---

## 七、必读论文

| 论文 | 关注章节 | 目的 |
|------|----------|------|
| 2404.14055v3 (RingID) | Sec 3 & 4.2 | 搞懂 Multi-Key 实现逻辑 |

---

## 八、时间节点

| 日期 | 任务 | 交付物 |
|------|------|--------|
| Day 1 (Mon) | 跑通 Demo | 确认环境可用 |
| Day 2 (Tue) | 生成 Multi-Key 数据 | 5 个 Key × 20 张图 |
| Day 2 (Tue) | 正交性测试 | 混淆矩阵数据 |
| Day 4 (Thu) | 撰写 Identifiability 章节 | Markdown/Word 文档 |

---

## 九、常见问题

### Q1: 模型下载失败
尝试切换 HuggingFace 端点，或让队友共享已下载的模型缓存

### Q2: GPU 显存不足
减少 `--num_images` 或 `--image_length`

### Q3: 生成速度太慢
减少 `--num_inference_steps`（默认 50，可改为 25）

---

**祝顺利！有问题随时问我。**

---

## 十、Prompt 数据集使用指南

### 10.1 数据集信息
- **名称**：`Gustavosta/Stable-Diffusion-Prompts`
- **来源**：HuggingFace Datasets
- **用途**：为 A 和 B 提供统一的 Prompt，确保实验可比性

### 10.2 代码已自动使用该数据集
在 `identify.py` 中，数据集已经被自动加载：
```python
# identify.py 第 79 行和第 107 行
dataset_id = 'Gustavosta/Stable-Diffusion-Prompts'
dataset, prompt_key = get_dataset(dataset_id)
```

`get_dataset()` 函数在 `utils.py` 第 426-439 行定义：
```python
def get_dataset(dataset):
    ...
    else:
        dataset = load_dataset(dataset)['test']
        prompt_key = 'Prompt'
    return dataset, prompt_key
```

### 10.3 手动获取 Prompt 列表（用于与 A 同学对齐）
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['test']

# 查看前 10 条 Prompt
for i in range(10):
    print(f"Prompt {i}: {dataset[i]['Prompt']}")

# 导出前 100 条 Prompt 到文件
with open('prompts_100.txt', 'w') as f:
    for i in range(100):
        f.write(f"{i}: {dataset[i]['Prompt']}\n")
```

### 10.4 离线使用（如果网络不通）
如果无法在线加载数据集，可以：

1. 在有网络的机器上运行：
```python
from datasets import load_dataset
dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')
dataset.save_to_disk('./sd_prompts_dataset')
```

2. 将 `sd_prompts_dataset` 文件夹拷贝到服务器

3. 修改代码使用本地数据集：
```python
from datasets import load_from_disk
dataset = load_from_disk('./sd_prompts_dataset')['test']
```

### 10.5 与 A 同学对齐的建议
1. 确认双方都使用 `Gustavosta/Stable-Diffusion-Prompts` 数据集
2. 确认使用的 Prompt 索引范围（如前 100 条）
3. 导出 `prompts_100.txt` 共享给 A，确保完全一致
