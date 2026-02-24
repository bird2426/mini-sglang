# Mini-SGLang 量化实现 Bug 分析报告

## 问题概述

Mini-SGLang 的量化实现**不完整，存在严重 bug**，无法正确运行量化模型。

## Bug 描述

### 问题 1：加载的量化权重与模型层不匹配

**位置**: `python/minisgl/engine/engine.py` 第 141-158 行

**问题描述**:
当配置了量化 (`config.quantization`) 时，`_load_weight_state_dict` 方法会调用 `load_quantized_weight` 加载量化权重（`qweight`, `scales`, `qzeros`）。但是，普通的 `Linear` 层（在 `layers/linear.py` 中定义）只有 `self.weight` 属性，没有 `self.qweight` 等量化权重属性。

**代码流程**:
```python
# engine.py line 52
self.model.load_state_dict(self._load_weight_state_dict(config))

# engine.py line 148-154
if config.quantization:
    state_dict, _ = load_quantized_weight(config.model_path, self.device)
    return {k: v.to(self.dtype) for k, v in state_dict.items()}
```

**结果**: 
- 量化模型的 state_dict 包含 `model.layers.0.self_attn.q_proj.qweight`
- 但 Linear 层期望的是 `model.layers.0.self_attn.q_proj.weight`
- 这会导致 `load_state_dict` 失败，报错 "Unexpected keys in state_dict"

### 问题 2：没有实现量化推理的计算层

**位置**: `python/minisgl/layers/linear.py`

**问题描述**:
SGLang (参考实现) 实现了专门的 `AWQLinearMethod` 类来处理量化推理：

```python
# SGLang 实现
class AWQLinearMethod(LinearMethodBase):
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias):
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        out = awq_dequantize(qweight, scales, qzeros)  # 量化权重解压缩
        out = torch.matmul(reshaped_x, out)           # 矩阵乘法
        ...
```

**Mini-SGLang 的情况**:
- 只有普通的 `LinearReplicated`, `LinearOProj`, `LinearRowParallel` 等类
- 这些类只实现了 `F.linear(x, self.weight, self.bias)`
- **完全没有**处理 `qweight`, `scales`, `qzeros` 的逻辑

### 问题 3：张量并行分割逻辑不完整

**位置**: `python/minisgl/quant/loader.py` 第 150-198 行

**问题描述**:
`shard_quantized_state_dict` 函数尝试对量化权重进行张量并行分割，但：

1. 只处理了基本的投影层 (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `o_proj`, `down_proj`)
2. 没有处理 MoE 层的专家权重
3. 分割逻辑与实际模型结构可能不匹配

## 根本原因

Mini-SGLang 的量化实现**只完成了配置加载部分**，**没有完成推理计算部分**。

当前的实现：
- ✅ 定义了 `AWQConfig` 配置类
- ✅ 实现了 `detect_quant_config` 自动检测
- ✅ 实现了 `load_quantized_weight` 权重加载
- ❌ 没有实现量化 Linear 层（需要处理 qweight/scales/qzeros）
- ❌ 没有实现量化权重的解压缩/推理计算
- ❌ 没有集成量化层到模型中

## 参考实现

参考 SGLang 的完整实现：

1. **量化配置层** (`sglang/srt/layers/quantization/awq.py`):
   - `AWQConfig`: 量化配置
   - `AWQLinearMethod`: 量化线性层计算方法
   - `AWQMarlinLinearMethod`: 优化后的 Marlin 量化方法

2. **权重处理**:
   - `create_weights`: 创建 qweight, scales, qzeros 参数
   - `process_weights_after_loading`: 加载后处理（格式转换）

3. **推理计算**:
   - `apply` 方法：执行 `x @ dequantize(weight) + bias`

4. **JIT Kernel**:
   - `awq_dequantize`: 量化权重解压缩 CUDA kernel
   - `awq_marlin_repack`: Marlin 格式重打包

## 修复方案

需要实现以下内容：

### 1. 添加量化 Linear 层

在 `python/minisgl/layers/linear.py` 中添加：

```python
class QuantizedLinear(BaseOP):
    """量化 Linear 层 (AWQ)"""
    
    def __init__(self, ...):
        self.qweight = torch.empty(...)
        self.scales = torch.empty(...)
        self.qzeros = torch.empty(...)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实现量化推理
        weight = dequantize(self.qweight, self.scales, self.qzeros)
        return F.linear(x, weight, self.bias)
```

### 2. 修改模型加载逻辑

在 `engine.py` 中：
- 检测量化配置
- 创建量化 Linear 层
- 加载量化权重到对应层

### 3. 添加 JIT Kernel（可选，用于性能优化）

参考 SGLang 实现 `awq_dequantize` kernel。

## 测试验证

修复后需要验证：
1. 加载 AWQ 量化模型不报错
2. 推理结果正确（与 FP16 基线对比）
3. 张量并行正常工作
4. 内存占用降低

## 相关文件

- `python/minisgl/quant/` - 量化配置和加载模块
- `python/minisgl/layers/linear.py` - 需要添加量化层
- `python/minisgl/engine/engine.py` - 需要修改模型加载逻辑
- `python/minisgl/models/` - 需要集成量化层到模型

## 时间线

- Bug 发现: 2026-02-24
