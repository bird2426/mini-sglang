# AWQ Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 mini-sglang 生产级 AWQ 4-bit 量化推理支持

**Architecture:** 
- 使用 Triton 实现 awq_dequantize kernel
- QuantizedLinear 层运行时解压缩（不是预解压缩）
- 扩展 loader.py 权重映射
- 修改 Qwen 模型支持量化配置

**Tech Stack:** Python, Triton, PyTorch

**Test Model:** Qwen/Qwen3-8B-AWQ-INT4

---

## Task 1: 添加 Triton awq_dequantize Kernel

**Files:**
- Create: `python/minisgl/kernel/awq_dequantize.py`

**Step 1: 创建 kernel 文件**

```python
# python/minisgl/kernel/awq_dequantize.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def awq_dequantize_kernel(
    qweight_ptr, scales_ptr, qzeros_ptr,
    output_ptr,
    M, N, K,  # M=out_features, N=in_features, K=group_size
    bits: tl.constexpr, group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # 实现 AWQ dequantization kernel
    # 从 int32 压缩权重解压缩为 fp16
    pass
```

**Step 2: 验证文件创建**

Run: `ls python/minisgl/kernel/awq_dequantize.py`

---

## Task 2: 实现 QuantizedLinear 层

**Files:**
- Modify: `python/minisgl/layers/linear.py`

**Step 1: 添加 QuantizedLinear 类**

```python
class QuantizedLinear(BaseOP):
    """Quantized Linear layer with on-the-fly dequantization"""
    
    def __init__(self, in_features, out_features, bits=4, group_size=128, has_bias=False):
        self.bits = bits
        self.group_size = group_size
        self.pack_factor = 32 // bits
        
        # 量化权重存储
        self.qweight = torch.empty(out_features // self.pack_factor, in_features, dtype=torch.int32)
        self.scales = torch.empty(in_features // group_size, out_features, dtype=torch.float16)
        self.qzeros = torch.empty(in_features // group_size, out_features // self.pack_factor, dtype=torch.int32)
        self.bias = torch.empty(out_features, dtype=torch.float16) if has_bias else None
    
    def forward(self, x):
        # 调用 Triton kernel 解压缩
        weight = awq_dequantize(self.qweight, self.scales, self.qzeros, 
                               bits=self.bits, group_size=self.group_size)
        return F.linear(x, weight, self.bias)
```

**Step 2: 验证导入**

Run: `cd .worktrees/awq-v2 && python -c "from minisgl.layers.linear import QuantizedLinear; print('OK')"`

---

## Task 3: 扩展 loader.py 权重映射

**Files:**
- Modify: `python/minisgl/quant/loader.py`

**Step 1: 添加权重映射函数**

```python
def create_quantized_weight_mapping(state_dict):
    """将 qweight/scales/qzeros 映射到层的量化参数"""
    mapping = {}
    for key, value in state_dict.items():
        if key.endswith('.qweight'):
            layer_name = key[:-len('.qweight')]
            mapping[f'{layer_name}.qweight'] = value
        elif key.endswith('.scales'):
            layer_name = key[:-len('.scales')]
            mapping[f'{layer_name}.scales'] = value
        elif key.endswith('.qzeros'):
            layer_name = key[:-len('.qzeros')]
            mapping[f'{layer_name}.qzeros'] = value
    return mapping
```

**Step 2: 验证语法**

Run: `python -m py_compile python/minisgl/quant/loader.py`

---

## Task 4: 修改 Qwen2 模型支持量化

**Files:**
- Modify: `python/minisgl/models/qwen2.py`

**Step 1: 检测量化配置并创建量化层**

```python
class Qwen2DecoderLayer:
    def __init__(self, config, layer_id, quant_config=None):
        # 根据 quant_config 创建 QuantizedLinear 或普通 Linear
        if quant_config and quant_config.is_layer_quantized(f'model.layers.{layer_id}.self_attn.q_proj'):
            self.self_attn.q_proj = QuantizedLinear(...)
        # ... 其他层
```

**Step 2: 验证模型创建**

Run: `python -c "from minisgl.models import create_model; print('OK')"`

---

## Task 5: 修改 Engine 集成量化流程

**Files:**
- Modify: `python/minisgl/engine/engine.py`

**Step 1: 修改模型创建和权重加载流程**

```python
def _load_weight_state_dict(self, config):
    if config.quantization:
        # 1. 检测量化配置
        # 2. 创建量化模型结构
        # 3. 加载量化权重到对应层
        ...
```

---

## Task 6: 测试验证

**Step 1: 下载测试模型**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-AWQ-INT4')"
```

**Step 2: 运行测试**

```bash
cd .worktrees/awq-v2
python -m minisgl --model Qwen/Qwen3-8B-AWQ-INT4 --shell
```

**预期:** 模型加载成功，可以进行对话

---

## Task 7: 提交代码

```bash
git add .
git commit -m "feat: implement AWQ quantization inference"
git push --set-upstream origin feature/awq-quantization-v2
```
