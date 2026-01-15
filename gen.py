import os
import random
import numpy as np
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# --- 全局配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 请根据实际情况修改模型路径
MODEL_ID = "D:/models/stable-diffusion-2-1-base" 
NUM_PROMPTS = 50
INFERENCE_STEPS = 50
SEED = 42
IMAGE_SIZE = 512
OUTPUT_DIR = "tree_ring_results"

# 水印配置：采用改进后的非对称双频带策略
# 低频用于抗压缩（强度低，半径小），高频用于隐蔽性（强度中，半径大）
WATERMARK_CONFIGS = {
    "Low-Freq": {"radius": 4, "alpha": 0.3, "dual": False},
    "High-Freq": {"radius": 18, "alpha": 0.5, "dual": False},
    "Dual-Ring": {
        "low_r": 4,         # 低频半径
        "high_r": 18,       # 高频半径
        "low_alpha": 0.2,   # 低频注入强度（较小以保护构图）
        "high_alpha": 0.5,  # 高频注入强度（适中以保证检测）
        "dual": True
    }
}

def set_random_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_prompts(num_prompts, prompt_file_path):
    """加载本地 Prompt，包含基础的错误处理和补充逻辑"""
    prompts = []
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    line = line.split(":", 1)[1].strip()
                if line:
                    prompts.append(line)
    
    if len(prompts) < num_prompts:
        default = "A cinematic shot of a mystical forest, highly detailed, 8k"
        prompts += [default] * (num_prompts - len(prompts))
    return prompts[:num_prompts]

def get_ring_mask(size, r_outer, r_inner=0, device="cpu"):
    """生成环形掩码，通过控制内径 r_inner 可以避开直流分量"""
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device), 
        torch.linspace(-1, 1, size, device=device), 
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)
    # 归一化半径坐标 (size/2 对应 1.0)
    mask = (dist <= (r_outer / (size / 2))) & (dist > (r_inner / (size / 2)))
    return mask

def inject_watermark(init_latents, config):
    """
    实现非对称双频注入逻辑：
    - 低频（Low-Freq）: 保护构图，提供鲁棒性
    - 高频（High-Freq）: 提供细节纹理中的隐蔽信息
    """
    device = init_latents.device
    dtype = init_latents.dtype
    batch_size, channels, height, width = init_latents.shape
    
    # 1. FFT 变换（提升到 float32 保证精度）
    latents_f32 = init_latents.to(torch.float32)
    latents_fft = torch.fft.fftshift(torch.fft.fft2(latents_f32, dim=(-2, -1)), dim=(-2, -1))
    
    # 2. 计算标准差以匹配能量分布
    latent_std = latents_fft.std(dim=(-2, -1), keepdim=True)
    
    # 3. 构造目标水印 Patch（固定随机种子）
    generator = torch.Generator(device=device).manual_seed(SEED)
    target_patch = torch.randn(latents_fft.shape, device=device, generator=generator) * latent_std
    target_patch = target_patch.to(torch.complex64)
    
    if config.get("dual", False):
        # --- 双频带注入 ---
        # A. 注入低频 (避开最中心 DC 分量 r_inner=1)
        mask_low = get_ring_mask(height, config["low_r"], r_inner=1, device=device)
        alpha_l = config["low_alpha"]
        latents_fft[:, :, mask_low] = (1 - alpha_l) * latents_fft[:, :, mask_low] + alpha_l * target_patch[:, :, mask_low]
        
        # B. 注入高频 (采用窄环带减少破坏)
        mask_high = get_ring_mask(height, config["high_r"], r_inner=config["high_r"]-2, device=device)
        alpha_h = config["high_alpha"]
        latents_fft[:, :, mask_high] = (1 - alpha_h) * latents_fft[:, :, mask_high] + alpha_h * target_patch[:, :, mask_high]
    else:
        # --- 单频带模式 ---
        r = config["radius"]
        alpha = config.get("alpha", 0.5)
        # 单频同样建议避开 DC 分量以保护画质
        mask = get_ring_mask(height, r, r_inner=1 if r > 1 else 0, device=device)
        latents_fft[:, :, mask] = (1 - alpha) * latents_fft[:, :, mask] + alpha * target_patch[:, :, mask]
    
    # 4. 逆变换回空域
    res_latents = torch.fft.ifft2(torch.fft.ifftshift(latents_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return res_latents.to(dtype)

def load_pipeline():
    print(f"正在从 {MODEL_ID} 加载模型...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe

def main():
    set_random_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    prompts = load_prompts(NUM_PROMPTS, "D:/models/prompts_for_alignment.txt")
    pipe = load_pipeline()

    for group_name, config in WATERMARK_CONFIGS.items():
        print(f"\n>>> 正在处理分组: {group_name}")
        group_path = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_path, exist_ok=True)
        
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            # 1. 生成原始 Latent
            latents_seed = SEED + i
            generator = torch.Generator(device=DEVICE).manual_seed(latents_seed)
            shape = (1, pipe.unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
            init_latents = torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float16)
            
            # 2. 注入水印
            watermarked_latents = inject_watermark(init_latents, config)
            
            # 3. 生成图像
            with torch.autocast(DEVICE):
                # 有水印
                out_w = pipe(
                    prompt=prompt, 
                    latents=watermarked_latents, 
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=7.5
                ).images[0]
                
                # 无水印
                out_no = pipe(
                    prompt=prompt, 
                    latents=init_latents, 
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=7.5
                ).images[0]
            
            # 4. 保存结果
            out_w.save(os.path.join(group_path, f"{i:03d}_watermarked.png"))
            out_no.save(os.path.join(group_path, f"{i:03d}_no_watermark.png"))

    print(f"\n生成完毕！结果保存在: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()