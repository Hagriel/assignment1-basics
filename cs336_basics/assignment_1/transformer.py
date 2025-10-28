"""Transformer architecture components.

Implements a decoder-only Transformer language model with:
- RMSNorm for layer normalization
- SwiGLU activation for feed-forward networks
- RoPE (Rotary Position Embedding) for positional encoding
- Multi-head self-attention with causal masking
- Pre-norm architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs using RMS statistics instead of mean and variance.
    More efficient than LayerNorm and commonly used in modern LLMs.

    Args:
        d_model: Dimensionality of input features
        eps: Small constant for numerical stability (default: 1e-5)
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # TODO: Initialize learnable scale parameter

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        raise NotImplementedError


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Implements the SwiGLU activation: SwiGLU(x, W, V, W2) = (Swish(xW) ⊙ xV)W2
    where Swish(x) = x * sigmoid(x) and ⊙ is element-wise multiplication.

    Used in models like PaLM and LLaMA instead of standard FFN with ReLU.

    Args:
        d_model: Input/output dimensionality
        d_ff: Hidden layer dimensionality (typically 4 * d_model)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # TODO: Initialize w1, w2, w3 linear layers

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        raise NotImplementedError


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Applies rotary positional embeddings to query and key tensors.
    RoPE encodes absolute positions with rotation matrices and has
    the property that relative position information is naturally encoded.

    Args:
        d_head: Dimensionality per attention head
        max_seq_len: Maximum sequence length to pre-compute frequencies for
        theta: Base for geometric progression (default: 10000.0)
    """

    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        # TODO: Pre-compute rotation frequencies

    def forward(
        self,
        x: Float[Tensor, "batch seq_len num_heads d_head"]
    ) -> Float[Tensor, "batch seq_len num_heads d_head"]:
        """Apply rotary position embeddings.

        Args:
            x: Input tensor (queries or keys)

        Returns:
            Tensor with rotary embeddings applied
        """
        raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE.

    Implements scaled dot-product attention across multiple heads:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        use_rope: Whether to use RoPE for positional encoding
        max_seq_len: Maximum sequence length (required if use_rope=True)
        theta: RoPE theta parameter (default: 10000.0)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float = 10000.0
    ) -> None:
        super().__init__()
        # TODO: Initialize query, key, value, and output projections
        # TODO: Initialize RoPE if use_rope=True

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor

        Returns:
            Attention output
        """
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block.

    Architecture (pre-norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimensionality
        max_seq_len: Maximum sequence length for RoPE
        theta: RoPE theta parameter
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0
    ) -> None:
        super().__init__()
        # TODO: Initialize attention, FFN, and normalization layers

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """Apply transformer block.

        Args:
            x: Input tensor

        Returns:
            Block output
        """
        raise NotImplementedError


class TransformerLM(nn.Module):
    """Transformer language model (decoder-only).

    Full transformer-based language model with:
    - Token embeddings
    - Multiple transformer blocks
    - Final layer norm
    - Language modeling head

    Args:
        vocab_size: Vocabulary size
        context_length: Maximum context length
        d_model: Model dimensionality
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Feed-forward hidden dimensionality
        rope_theta: RoPE theta parameter
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0
    ) -> None:
        super().__init__()
        # TODO: Initialize embeddings, transformer blocks, final norm, and LM head

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        """Forward pass through the language model.

        Args:
            input_ids: Token indices

        Returns:
            Unnormalized next-token logits
        """
        raise NotImplementedError