# RingID 项目代码详解

## 一、项目概述

### 1.1 什么是 RingID？
RingID 是一种基于 **傅里叶空间** 的扩散模型图像水印技术，是 Tree-Ring 的增强版本。

**核心改进**：
- Tree-Ring：只能判断"有/无水印"（二分类）
- RingID：可以识别"是哪个供应商的水印"（多分类）

### 1.2 核心思想
在生成图像时，将特定的"指纹"注入到图像的**傅里叶频域**中。这个指纹：
- 人眼不可见
- 对各种攻击（压缩、裁剪、模糊）有一定鲁棒性
- 不同供应商使用不同的指纹（Key）

---

## 二、项目结构

```
RingID/
├── identify.py          # 主程序：Multi-Key 识别实验
├── verify.py            # 单 Key 验证实验
├── utils.py             # 核心工具函数
├── inverse_stable_diffusion.py  # 可逆 SD Pipeline
├── modified_stable_diffusion.py # 修改版 SD Pipeline
├── optim_utils.py       # 优化相关工具
├── io_utils.py          # 输入输出工具
├── open_clip/           # CLIP 模型（用于评估图像质量）
└── guided_diffusion/    # 扩散模型相关
```

---

## 三、核心流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        生成阶段 (Generation)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. 生成随机噪声 z₀                                                │
│         ↓                                                         │
│  2. 在傅里叶空间注入水印 → z₀' = z₀ + watermark_pattern           │
│         ↓                                                         │
│  3. 通过 Stable Diffusion 生成图像                                 │
│         ↓                                                         │
│  4. 输出带水印的图像                                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        检测阶段 (Detection)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. 输入待检测图像                                                 │
│         ↓                                                         │
│  2. 通过 DDIM Inversion 反推初始噪声 z₀'                          │
│         ↓                                                         │
│  3. 对 z₀' 做 FFT，提取傅里叶空间特征                              │
│         ↓                                                         │
│  4. 与所有已知 Key 的 pattern 计算距离                             │
│         ↓                                                         │
│  5. 距离最小的 Key = 预测结果                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、关键代码详解

### 4.1 水印参数 (`utils.py` 第 15-23 行)

```python
RADIUS = 14              # 水印环的外半径
RADIUS_CUTOFF = 3        # 水印环的内半径（中心被截断）
ANCHOR_X_OFFSET = 0      # 锚点 X 偏移
ANCHOR_Y_OFFSET = 0      # 锚点 Y 偏移

HETER_WATERMARK_CHANNEL = [0]   # 异构水印通道
RING_WATERMARK_CHANNEL = [3]    # 环形水印通道
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)
```

**解释**：
- 水印被注入到 latent 空间的**特定通道**（通道 0 和 3）
- 水印区域是一个"圆环"，外半径 14，内半径 3
- 这意味着水印占据半径 3-14 的环形区域

### 4.2 环形掩码生成 (`utils.py` 第 121-124 行)

```python
def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF, ...):
    outer_mask = circle_mask(size=size, r=r_out, ...)
    inner_mask = circle_mask(size=size, r=r_in, ...)
    return outer_mask & (~inner_mask)  # 外圆 - 内圆 = 圆环
```

**解释**：
- 生成一个 64×64 的布尔掩码
- `True` 的位置是水印区域（圆环）
- `False` 的位置保持原始噪声

### 4.3 Multi-Key 生成 (`identify.py` 第 148-161 行)

```python
# 计算每个环可以取的值
key_value_list = [
    [list(combo) for combo in itertools.product(
        np.linspace(-args.ring_value_range, args.ring_value_range, args.num_inmost_keys).tolist(),
        repeat=len(RING_WATERMARK_CHANNEL)
    )]
    for _ in range(single_channel_num_slots)
]

# 生成所有可能的 Key 组合
key_value_combinations = list(itertools.product(*key_value_list))

# 随机选择指定数量的 Key
if args.assigned_keys > 0:
    key_value_combinations = random.sample(key_value_combinations, k=args.assigned_keys)
```

**解释**：
- `single_channel_num_slots = RADIUS - RADIUS_CUTOFF = 11`（11 个环）
- 每个环可以取 `num_inmost_keys` 个不同的值
- 不同的值组合 = 不同的 Key
- 理论上可以生成 `num_inmost_keys^11` 个不同的 Key

### 4.4 水印 Pattern 生成 (`utils.py` 第 266-323 行)

```python
def make_Fourier_ringid_pattern(device, key_value_combination, no_watermark_latents, ...):
    # 初始化空的水印 pattern
    watermarked_latents_fft = torch.zeros_like(latents_fft)
    
    # 遍历每个环
    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        
        # 生成该环的掩码
        mask = torch.tensor(ring_mask(size=shape[-1], r_out=this_r_out, r_in=this_r_in))
        
        # 在该环上填充对应的 Key 值
        for channel_index in range(len(ring_watermark_channel)):
            watermarked_latents_fft[..., ring_watermark_channel[channel_index]].real = \
                (1 - mask) * ... + mask * key_value_combination[radius_index][channel_index]
            watermarked_latents_fft[..., ring_watermark_channel[channel_index]].imag = \
                (1 - mask) * ... + mask * key_value_combination[radius_index][channel_index]
    
    return watermarked_latents_fft
```

**解释**：
- 每个 Key 由 11 个环的值组成
- 每个环填充一个特定的值（来自 `key_value_combination`）
- 同时修改实部和虚部

### 4.5 水印注入 (`utils.py` 第 184-205 行)

```python
def generate_Fourier_watermark_latents(device, radius, radius_cutoff, 
                                        watermark_region_mask, watermark_channel,
                                        original_latents, watermark_pattern):
    # 对原始噪声做 FFT
    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim=(-1, -2))
    
    # 在水印区域替换为水印 pattern
    for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
        watermarked_latents_fft[:, channel] = \
            watermarked_latents_fft[:, channel] * ~channel_mask + \
            watermark_pattern[:, channel] * channel_mask
    
    # 做 IFFT 回到空间域
    return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim=(-1, -2))).real
```

**解释**：
1. 对原始噪声 `z₀` 做 2D FFT，转到频域
2. 在圆环区域，用水印 pattern 替换原始值
3. 做 IFFT 转回空间域，得到带水印的噪声 `z₀'`

### 4.6 检测与识别 (`identify.py` 第 247-266 行)

```python
# DDIM Inversion：从图像反推初始噪声
Fourier_watermark_reconstructed_latents = pipe.forward_diffusion(
    latents=Fourier_watermark_image_latents,
    text_embeddings=text_embeddings,
    guidance_scale=1,
    num_inference_steps=args.test_num_inference_steps,
)

# 对反推的噪声做 FFT
Fourier_watermark_reconstructed_latents_fft = fft(Fourier_watermark_reconstructed_latents)

# 与所有 Key 的 pattern 计算距离
for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
    distance_per_gt = get_distance(
        Fourier_watermark_pattern, 
        single_rec_latent_fft[None, ...],
        watermark_region_mask, 
        channel=WATERMARK_CHANNEL,
        ...
    )
    distances_list.append(distance_per_gt)

# 距离最小的 Key = 预测结果
acc = np.argmin(np.array(distances_list)) == key_index
```

**解释**：
1. 用 DDIM Inversion 从图像反推初始噪声
2. 对噪声做 FFT
3. 计算与每个已知 Key 的 L1 距离
4. 距离最小的 Key 就是预测的供应商

### 4.7 距离计算 (`utils.py` 第 325-386 行)

```python
def get_distance(tensor1, tensor2, mask, p, mode, ...):
    if p == 1:
        if mode == 'complex':
            # L1 距离：|a - b| 的平均值
            return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
```

**解释**：
- 只在水印区域（mask=True）计算距离
- 使用 L1 距离（绝对值差的平均）
- 距离越小，说明越匹配

---

## 五、攻击与鲁棒性 (`utils.py` 第 208-256 行)

```python
def image_distortion(img1, img2, seed, 
                     r_degree=None,        # 旋转
                     jpeg_ratio=None,      # JPEG 压缩
                     crop_scale=None,      # 裁剪
                     gaussian_blur_r=None, # 高斯模糊
                     gaussian_std=None,    # 高斯噪声
                     brightness_factor=None):  # 亮度调整
```

**支持的攻击类型**：
| 攻击 | 参数 | 说明 |
|------|------|------|
| 旋转 | `r_degree=75` | 旋转 75 度 |
| JPEG 压缩 | `jpeg_ratio=25` | 质量 25% |
| 裁剪 | `crop_scale=0.75` | 裁剪到 75% |
| 高斯模糊 | `gaussian_blur_r=8` | 半径 8 |
| 高斯噪声 | `gaussian_std=0.1` | 标准差 0.1 |
| 亮度 | `brightness_factor=6` | 亮度因子 6 |

---

## 六、命令行参数详解

```bash
python identify.py \
    --run_name multi_key_test \    # 实验名称
    --online \                      # 在线下载模型
    --trials 100 \                  # 生成 100 张图
    --assigned_keys 5 \             # 使用 5 个不同的 Key
    --num_images 1 \                # 每个 prompt 生成 1 张图
    --save_generated_imgs 1 \       # 保存生成的图片
    --num_inference_steps 50 \      # 推理步数
    --image_length 512 \            # 图像尺寸
    --ring_value_range 64 \         # Key 值范围 [-64, 64]
    --num_inmost_keys 2             # 每个环可取 2 个值
```

---

## 七、输出解读

### 7.1 终端输出
```
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
|     Id Acc     | Clean | Rot 75 | JPEG 25 | C&S 75 | Blur 8 | Noise 0.1 | Brightness... |  Avg  |
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
| L1 |a-b|       | 0.950 | 0.820  | 0.780   | 0.850  | 0.900  |   0.880   |     0.750     | 0.847 |
+----------------+-------+--------+---------+--------+--------+-----------+---------------+-------+
```

- **Clean**：无攻击时的识别准确率
- **Rot 75**：旋转 75 度后的准确率
- **JPEG 25**：JPEG 压缩 25% 后的准确率
- **Avg**：平均准确率

### 7.2 输出文件
- `runs/xxx/images/watermarked/`：带水印的图片
- `runs/xxx/images/no_watermark/`：无水印的图片
- `runs/xxx/log.csv`：详细的实验日志

---

## 八、关键公式

### 8.1 傅里叶变换
```
F(u,v) = FFT(f(x,y))
```
将空间域的图像/噪声转换到频域。

### 8.2 水印注入
```
Z'(u,v) = Z(u,v) * (1 - M) + W(u,v) * M
```
- `Z`：原始噪声的频域表示
- `M`：圆环掩码
- `W`：水印 pattern
- `Z'`：带水印的噪声

### 8.3 检测距离
```
d(K_i, Z') = mean(|W_i - Z'| * M)
```
- 在掩码区域计算 L1 距离
- 距离最小的 Key 即为预测结果

---

## 九、快速上手

### 9.1 最小可运行示例
```bash
conda activate tr
python identify.py --run_name quick_test --online --trials 5 --assigned_keys 2
```

### 9.2 生成 5 Key × 20 图（你的任务）
```bash
python identify.py \
    --run_name multi_key_5vendors \
    --online \
    --trials 100 \
    --assigned_keys 5 \
    --save_generated_imgs 1
```

---

**有问题随时问我！**
