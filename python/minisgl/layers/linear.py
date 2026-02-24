from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = div_even(num_qo_heads, num_kv_heads)
        local_num_kv = div_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize AWQ quantized weights.

    Args:
        qweight: Quantized weight tensor of shape [in_features, out_features // pack_factor]
            stored as packed int32 (each int32 contains pack_factor weights)
        scales: Scale tensor of shape [num_groups, out_features]
        qzeros: Quantized zero points tensor of shape [num_groups, out_features // pack_factor]
        bits: Number of bits per weight (4 or 8)
        group_size: Group size for quantization

    Returns:
        Dequantized weight tensor of shape [in_features, out_features]
    """
    pack_factor = 32 // bits
    out_features = qweight.shape[1] * pack_factor
    in_features = qweight.shape[0] * group_size

    # Reshape for dequantization
    num_groups = qweight.shape[0] // group_size

    # Unpack weights from int32 to individual bits
    # For 4-bit: each int32 contains 8 values
    # qweight shape: [in_features // group_size, out_features // pack_factor]
    # After reshape: [num_groups, group_size, out_features // pack_factor]
    qweight_reshaped = qweight.view(num_groups, group_size, -1)
    qzeros_reshaped = qzeros.view(num_groups, -1)

    # Extract 4-bit values from int32
    # Each int32 can store 8 4-bit values
    # Unpack using bit shifts and masks
    unpacked = torch.zeros(
        num_groups, group_size, out_features, dtype=scales.dtype, device=qweight.device
    )

    for i in range(pack_factor):
        shift = i * bits
        mask = (1 << bits) - 1
        # Extract the i-th 4-bit value from each int32
        unpacked[:, :, i::pack_factor] = ((qweight_reshaped >> shift) & mask).to(scales.dtype)

    # Dequantize: w = (w_q - zero) * scale
    # Expand scales to match unpacked shape
    scales_expanded = scales.unsqueeze(1).expand(num_groups, group_size, out_features)
    qzeros_expanded = qzeros.unsqueeze(1).expand(num_groups, group_size, out_features)

    # Apply dequantization
    dequantized = (unpacked - qzeros_expanded) * scales_expanded

    # Reshape to [in_features, out_features]
    dequantized = dequantized.view(in_features, out_features)

    return dequantized


class QuantizedLinear(BaseOP):
    """Quantized Linear layer for AWQ inference.

    This layer stores weights in quantized format (qweight, scales, qzeros)
    and dequantizes on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        has_bias: bool = False,
    ):
        """Initialize quantized linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bits: Number of bits for quantization (4 or 8)
            group_size: Group size for quantization
            has_bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.pack_factor = 32 // bits
        self.has_bias = has_bias

        # Calculate quantized dimensions
        self.num_groups = div_even(in_features, group_size)
        self.qweight_shape = (self.num_groups * group_size, out_features // self.pack_factor)
        self.scales_shape = (self.num_groups, out_features)
        self.qzeros_shape = (self.num_groups, out_features // self.pack_factor)

        # Allocate quantized weight storage
        self.qweight = torch.empty(self.qweight_shape[1], self.qweight_shape[0], dtype=torch.int32)
        self.scales = torch.empty(self.scales_shape, dtype=torch.float16)
        self.qzeros = torch.empty(self.qzeros_shape[1], self.qzeros_shape[0], dtype=torch.int32)
        self.bias = torch.empty(out_features, dtype=torch.float16) if has_bias else None

    def load_quantized_weight(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Load quantized weights from pre-quantized data.

        Args:
            qweight: Quantized weight [out_features // pack_factor, in_features]
            scales: Scale factors [num_groups, out_features]
            qzeros: Quantized zero points [out_features // pack_factor, num_groups]
            bias: Optional bias [out_features]
        """
        # Transpose to match internal layout
        # External: qweight [out_features // pack_factor, in_features]
        # Internal: qweight [out_features // pack_factor, num_groups * group_size]
        self.qweight = qweight.t().contiguous()
        self.scales = scales.t().contiguous()
        self.qzeros = qzeros.t().contiguous()

        if bias is not None:
            if self.bias is None:
                self.bias = torch.empty(self.out_features, dtype=torch.float16)
            self.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        # Transpose qweight back to [in_features, out_features // pack_factor] for dequantize
        qweight_t = self.qweight.t()  # [out_features // pack_factor, in_features]
        qzeros_t = self.qzeros.t()  # [out_features // pack_factor, num_groups]
        scales_t = self.scales.t()  # [out_features, num_groups]

        weight = awq_dequantize(
            qweight_t,
            scales_t,
            qzeros_t,
            bits=self.bits,
            group_size=self.group_size,
        )

        return F.linear(x, weight, self.bias)


class QuantizedLinearFP8(BaseOP):
    """FP8 Quantized Linear layer (simpler format).

    This is a simpler FP8 quantization where weights are stored as float8_e4m3fn.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        self.weight = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(out_features, dtype=torch.float16)
        self.bias = torch.empty(out_features, dtype=torch.float16) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize to float16 for computation
        weight = self.weight.to(torch.float16) * self.scale.unsqueeze(1)
        return F.linear(x, weight, self.bias)
