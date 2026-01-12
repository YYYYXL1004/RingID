# RingID 问题排查记录

## 问题 1：网络不通，无法下载 HuggingFace 模型

### 错误信息
```
OSError: [Errno 101] Network is unreachable
huggingface_hub.utils._errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub
```

### 原因
服务器无法访问 huggingface.co

### 解决方法
使用 ModelScope 下载模型到本地：
```bash
conda activate tr
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/stable-diffusion-2-1-base', cache_dir='./models')"
```

模型会下载到 `./models/AI-ModelScope/stable-diffusion-2-1-base/`

---

## 问题 2：CLIPImageProcessor 找不到

### 错误信息
```
AttributeError: module transformers has no attribute CLIPImageProcessor
```

### 原因
`transformers==4.23.1` 版本太旧，没有 `CLIPImageProcessor` 类。这个类在 `transformers>=4.26.0` 才引入。

但 RingID 代码要求 `diffusers==0.11.1`，而这个版本的 diffusers 与新版 transformers 不兼容。

### 解决方法

**方法 1：升级 transformers（推荐）**
```bash
conda activate tr
pip install transformers==4.26.0
```

然后重新运行：
```bash
python identify.py --run_name test_demo --trials 5 --assigned_keys 3
```

**方法 2：如果方法 1 导致其他兼容性问题**

降级 transformers 并修改模型配置：
```bash
# 检查模型配置文件
cat ./models/AI-ModelScope/stable-diffusion-2-1-base/feature_extractor/preprocessor_config.json
```

如果里面有 `"image_processor_type": "CLIPImageProcessor"`，改成 `"feature_extractor_type": "CLIPFeatureExtractor"`

**方法 3：创建新环境（最干净）**
```bash
conda create -n ringid python=3.10
conda activate ringid
pip install torch==1.13.0 torchvision
pip install diffusers==0.11.1
pip install transformers==4.26.0
pip install datasets ftfy matplotlib accelerate scikit-image scikit-learn prettytable open_clip_torch
```

---

## 问题 3：CLIP 模型权重找不到

### 错误信息
```
RuntimeError: Pretrained weights (/sda/home/yaoxiangling/.cache/huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/.../open_clip_pytorch_model.bin) not found for model ViT-g-14.
Available pretrained tags (['laion2b_s12b_b42k', 'laion2b_s34b_b88k'].
```

### 原因
代码在离线模式下尝试加载本地缓存的 CLIP 模型，但该文件不存在。

### 解决方法

**方法 1：使用在线模式下载 CLIP（需要网络）**
```bash
# 如果有网络
python identify.py --run_name test_demo --online --trials 5 --assigned_keys 3
```

**方法 2：跳过 CLIP 模型（推荐，不影响核心功能）**

CLIP 模型只用于计算图像质量指标（CLIP Score），不影响水印生成和识别。

运行时添加 `--reference_model None`：
```bash
python identify.py --run_name test_demo --trials 5 --assigned_keys 3 --reference_model None
```

**方法 3：使用 ModelScope 下载 CLIP 模型**
```bash
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/CLIP-ViT-g-14-laion2B-s12B-b42K', cache_dir='./models')"
```

然后修改 `identify.py` 第 84 行的 `reference_model_pretrain` 路径。

---

## 问题 4：`--reference_model None` 参数无效

### 错误信息
```
RuntimeError: Model config for None not found.
```

### 原因
`--reference_model None` 被解析为字符串 `"None"` 而不是 Python 的 `None`，代码仍然尝试加载名为 "None" 的模型。

### 解决方法

需要修改 `identify.py` 代码，让它正确处理 `None` 参数。

**修改 `identify.py` 第 99-105 行**：

将：
```python
if args.reference_model is not None:
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
```

改为：
```python
if args.reference_model is not None and args.reference_model.lower() != 'none':
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
```

或者直接运行时不传 `--reference_model` 参数，而是修改代码默认跳过 CLIP。

**快速修复命令**（由 AI 助手执行）：
让 AI 助手帮你修改代码，添加对 `None` 字符串的处理。

---

## 问题 5：无法加载 Prompt 数据集（网络问题）

### 错误信息
```
ConnectionError: Couldn't reach 'Gustavosta/stable-diffusion-prompts' on the Hub (ConnectionError)
```

### 原因
代码尝试从 HuggingFace Hub 加载 Prompt 数据集，但网络不通。

### 解决方法

**方法 1：使用本地 Prompt 文件（推荐）**

创建一个本地 Prompt 文件，然后修改代码使用它。

1. 创建 `prompts.txt` 文件，每行一个 Prompt：
```
a beautiful sunset over the ocean
a cat sitting on a windowsill
a futuristic city at night
a portrait of a woman with flowers
a mountain landscape with snow
```

2. 修改 `utils.py` 中的 `get_dataset()` 函数，添加本地文件支持。

**方法 2：在有网络的机器上下载数据集**

在有网络的机器上运行：
```python
from datasets import load_dataset
dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')
dataset.save_to_disk('./sd_prompts_dataset')
```

然后把 `sd_prompts_dataset` 文件夹拷贝到服务器。

**方法 3：修改代码使用硬编码 Prompt 列表**

修改 `identify.py`，直接使用一个 Prompt 列表而不是从数据集加载。

**快速修复**：让 AI 助手帮你修改代码，创建本地 Prompt 列表。

---

## 问题 6：（预留）

### 错误信息
```
（待填写）
```

### 解决方法
（待填写）

---

## 常用调试命令

### 检查环境版本
```bash
python -c "import torch; import diffusers; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'diffusers: {diffusers.__version__}'); print(f'transformers: {transformers.__version__}')"
```

### 检查模型文件
```bash
ls -la ./models/AI-ModelScope/stable-diffusion-2-1-base/
```

### 检查 GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```
