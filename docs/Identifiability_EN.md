# Multi-Key Orthogonality Analysis from a Frequency Band Decoupling Perspective

> **Author**: Member B  
> **Role**: The Challenger  
> **Core Contributions**: Frequency-band encoding orthogonality analysis + Capacity limit exploration

---

## 1. Research Motivation

### 1.1 Connection to Project Theme

The core theme of this project is **"Evaluation and Optimization of Generative Watermarking Technology Based on Frequency Band Decoupling"**. While Member A explores watermark injection strategies across different frequency bands from a "robustness" perspective, this study analyzes how frequency-domain encoding achieves multi-key differentiation from an **"identifiability"** perspective.

**Two Dimensions of Frequency Band Decoupling**:

| Dimension | Author | Research Question |
|-----------|--------|-------------------|
| **Robustness** | Member A | How do watermarks in different frequency bands resist attacks? |
| **Identifiability** | Member B | How does frequency-band encoding achieve key differentiation? |

### 1.2 Problem Statement

Traditional Tree-Ring watermarking can only answer a simple binary question: "Does this image contain a watermark?" However, in multi-vendor scenarios, we need to answer:

> **"Which vendor generated this image?"**

RingID achieves multi-key differentiation by encoding different values on **different radius rings** (i.e., different frequency bands) in the Fourier frequency domain. This directly relates to our project's "frequency band decoupling" theme.

### 1.3 Research Objectives

From a frequency band decoupling perspective, this study comprehensively evaluates RingID's multi-key identification capability:

1. **Key Orthogonality Analysis**: Verify the interference level between keys encoded in different frequency bands
2. **Capacity Limit Exploration** 🌟: Explore the trade-off between frequency-band encoding capacity and identification accuracy

---

## 2. Technical Approach

### 2.1 Frequency-Band Encoding and Key Differentiation

The core idea of RingID is to use **different radius rings** (corresponding to different frequency bands) in the Fourier frequency domain for key encoding. This complements Member A's "dual-band strategy":

| Study | Frequency Band Usage | Goal |
|-------|---------------------|------|
| Member A | Low-freq vs High-freq vs Dual-band | Improve robustness |
| Member B | Combination encoding of 11 frequency bands | Achieve multi-key differentiation |

```
Frequency-Band Encoding Illustration:

        ┌───────────────────┐
        │    ○ ○ ○ ○ ○    │  ← High-frequency band (R=14)
        │  ○           ○  │
        │ ○     ●●●     ○ │  ← Low-frequency band (R=3)
        │  ○           ○  │
        │    ○ ○ ○ ○ ○    │
        └───────────────────┘
        
        Encoding region: 11 concentric rings (R=3 to R=14)
        Each ring = 1 frequency band = 1 encoding bit
        Theoretical key capacity: K^11
```

**Physical Meaning of Frequency-Band Encoding**:
- **Low-frequency rings** (R=3-7): Correspond to low-frequency components, strong resistance to compression attacks
- **High-frequency rings** (R=8-14): Correspond to high-frequency components, minimal impact on image quality
- **Combination encoding**: Different frequency band values combine to form unique keys

**Core Parameters**:
- Encoding radius range: $R_{in}=3$ to $R_{out}=14$
- Number of frequency bands: $11$ independent encoding bits
- Theoretical capacity: $K^{11}$ distinct keys ($2048$ when $K=2$)

### 2.2 Identification Algorithm

Given a test image $I$ and $N$ candidate keys $\{K_1, K_2, ..., K_N\}$, the identification process is:

1. Perform DDIM inversion on the image to obtain latent representation $z$
2. Apply 2D FFT to $z$ to extract frequency domain features $F(z)$
3. Compute distance to each key template: $d_i = ||F(z) - T_{K_i}||_1$
4. Select the key with minimum distance: $\hat{K} = \arg\min_i d_i$

---

## 3. Experimental Design

### 3.1 Basic Identification Experiment

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of Keys | 5 | Simulating 5 different vendors |
| Images per Key | 20 | Each vendor generates 20 images |
| Total Images | 100 | 5 × 20 = 100 |
| Image Resolution | 512×512 | Standard SD 2.1 output |
| Inference Steps | 50 | DDIM sampling steps |
| Prompt Source | Gustavosta/Stable-Diffusion-Prompts | Aligned with team members |

### 3.2 Key Capacity Limit Test 🌟

**Innovation**: Systematically explore the relationship between key quantity and identification accuracy.

| Number of Keys | Test Images | Images per Key |
|----------------|-------------|----------------|
| 5 | 50 | 10 |
| 10 | 100 | 10 |
| 20 | 200 | 10 |
| 30 | 300 | 10 |
| 50 | 500 | 10 |

### 3.3 Attack Scenarios

Testing 7 common image attacks:

| Attack Type | Parameters | Real-world Scenario |
|-------------|------------|---------------------|
| Clean | - | No attack baseline |
| Rotation | 75° | Image editing rotation |
| JPEG | Q=25 | Social media compression |
| Crop & Scale | 75% | Screenshot and enlarge |
| Gaussian Blur | r=8 | Blur processing |
| Gaussian Noise | σ=0.1 | Noise interference |
| Brightness | [0,6] | Brightness adjustment |

### 3.4 Evaluation Metrics

**Identification Accuracy**:
$$\text{Acc} = \frac{\sum_{i=1}^{N} \mathbb{1}[\hat{K}_i = K_i^*]}{N}$$

**Confusion Matrix**: Visualize misidentification patterns between keys

---

## 4. Experimental Results

### 4.1 Basic Identification Results (5 Keys)

| Attack Type | Accuracy | Rating |
|-------------|----------|--------|
| Clean | **100.0%** | ★★★★★ |
| Rotation 75° | **100.0%** | ★★★★★ |
| JPEG 25% | **100.0%** | ★★★★★ |
| Crop & Scale 75% | 90.0% | ★★★☆☆ |
| Gaussian Blur 8 | **100.0%** | ★★★★★ |
| Gaussian Noise 0.1 | **100.0%** | ★★★★★ |
| Brightness | 98.0% | ★★★★☆ |
| **Average** | **98.3%** | ★★★★★ |

**CLIP Quality Assessment**:
- No-watermark image CLIP Score: 0.368
- Watermarked image CLIP Score: 0.355
- Quality degradation: **3.5%** (negligible)

### 4.2 Key Capacity Limit Test Results 🌟

| Keys | Clean | Rot75 | JPEG25 | C&S75 | Blur8 | Noise | Bright | **Avg** |
|------|-------|-------|--------|-------|-------|-------|--------|---------|
| **5** | 100% | 100% | 100% | 90.0% | 100% | 100% | 98.0% | **98.3%** |
| **10** | 100% | 100% | 100% | 78.0% | 100% | 100% | 97.0% | **96.4%** |
| **20** | 100% | 100% | 100% | 71.0% | 100% | 100% | 96.5% | **95.4%** |
| **30** | 100% | 99.7% | 100% | 58.3% | 100% | 98.7% | 97.0% | **93.4%** |
| **50** | 100% | 98.2% | 99.8% | 48.4% | 99.6% | 98.0% | 98.0% | **91.7%** |

**Key Findings**:
- As keys increase from 5 to 50, average accuracy drops from 98.3% to 91.7%
- **Crop & Scale attack is the main bottleneck**: drops from 90% to 48.4%
- Under other attack scenarios, even 50 keys maintain 98%+ accuracy

### 4.3 Capacity-Accuracy Curve

![Key Capacity-Accuracy Curve](../runs/key_capacity_test/key_capacity_curve.png)

**Curve Analysis**:
- **Safe Zone** (5-20 Keys): Average accuracy > 95%, suitable for most applications
- **Transition Zone** (20-30 Keys): Accuracy begins to decline noticeably
- **Risk Zone** (30+ Keys): Crop & Scale accuracy falls below 60%

---

## 5. In-depth Analysis

### 5.1 Key Orthogonality Analysis

The confusion matrix shows good orthogonality between keys:

```
              Predicted Key
              K0    K1    K2    K3    K4
Actual Key K0 [████]  [  ]  [  ]  [  ]  [  ]   98%
           K1 [  ]  [████]  [  ]  [  ]  [  ]   97%
           K2 [  ]  [  ]  [████]  [  ]  [  ]   99%
           K3 [  ]  [  ]  [  ]  [████]  [  ]   98%
           K4 [  ]  [  ]  [  ]  [  ]  [████]   99%
```

- **Diagonal elements**: Average 98.2% (correct identification)
- **Off-diagonal elements**: Average < 2% (misidentification)
- **Maximum misidentification**: Slightly higher between adjacent keys

### 5.2 Attack Robustness Analysis

| Attack Type | Robustness | Reason |
|-------------|------------|--------|
| JPEG Compression | ★★★★★ | Frequency domain watermark in low-frequency region, unaffected by DCT |
| Rotation | ★★★★★ | Ring structure has rotational invariance |
| Blur/Noise | ★★★★☆ | Low-pass filtering has limited impact on ring region |
| Crop & Scale | ★★☆☆☆ | Destroys ring structure in frequency domain |

### 5.3 Comparison from Frequency Band Decoupling Perspective

| Feature | Tree-Ring | RingID | Frequency Band Perspective |
|---------|-----------|--------|---------------------------|
| Task Type | Binary | **Multi-class** | Multi-band encoding enables key differentiation |
| Key Capacity | 1 | **$K^{11}$** | Combination encoding of 11 frequency bands |
| Band Utilization | Single band | **Multi-band combination** | Complements Member A's dual-band strategy |
| Practical Key Count | 1 | **20-30** | Practical limit of frequency band capacity |

---

## 6. Conclusions and Recommendations

### 6.1 Main Conclusions

1. **Frequency-Band Encoding Effectiveness**: Combination encoding based on 11 frequency bands achieves **98.3%** identification accuracy in 5-key scenarios
2. **Key Orthogonality**: Minimal interference between keys encoded in different frequency bands, misidentification rate below **2%**
3. **Frequency Band Capacity Limit** 🌟: Practical key capacity is **20-30**; beyond this range, inter-band interference increases
4. **Connection to Robustness**: Crop & Scale attacks destroy frequency domain structure, representing the main bottleneck for frequency-band encoding

### 6.2 Application Recommendations

| Scenario | Recommended Keys | Expected Accuracy |
|----------|------------------|-------------------|
| Small Teams (< 10 vendors) | 5-10 | > 96% |
| Medium Platforms (10-20 vendors) | 10-20 | > 95% |
| Large Platforms (20+ vendors) | 20-30 | > 93% |

### 6.3 Connection to Overall Project

This study's connection to the project theme "Evaluation and Optimization of Generative Watermarking Technology Based on Frequency Band Decoupling":

| Dimension | Member A | Member B | Complementary Relationship |
|-----------|----------|----------|---------------------------|
| Research Question | Impact of band selection on robustness | Impact of band encoding on identifiability | Robustness vs Identifiability |
| Core Finding | Dual-band strategy improves robustness | Multi-band encoding enables key differentiation | Two applications of frequency band decoupling |
| Limitation | High-frequency watermarks weak against attacks | Frequency band capacity has upper limit | Common frequency domain constraints |

### 6.4 Limitations and Future Work

- **Crop & Scale Attack**: Destroys frequency domain ring structure; multi-scale frequency-band encoding could be explored
- **Frequency Band Capacity Expansion**: Multi-channel encoding could further expand key space
- **Combining with Dual-Band Strategy**: Explore maximizing key capacity while ensuring robustness

---

## Appendix

### A. Experimental Environment

| Item | Configuration |
|------|---------------|
| GPU | NVIDIA RTX 4090 |
| Model | Stable Diffusion 2.1 Base |
| Framework | PyTorch 1.13.0 + Diffusers 0.11.1 |
| Evaluation Model | CLIP ViT-g-14 (laion2b_s12b_b42k) |

### B. Experimental Commands

```bash
# Basic identification experiment
python identify.py \
    --run_name multi_key_5vendors_v2 \
    --trials 100 \
    --assigned_keys 5 \
    --save_generated_imgs 1 \
    --gpu_id 3

# Key capacity test
python scripts/key_capacity_test.py --keys 5,10,20,30,50 --gpu 3
```

### C. Data Alignment

- Prompt source: First 500 entries from `Gustavosta/Stable-Diffusion-Prompts`
- Random seed: `--general_seed 0`
- Alignment file: `prompts_for_alignment.txt`
