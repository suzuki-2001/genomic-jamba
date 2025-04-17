import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mamba_ssm import Mamba2
from flash_attn import flash_attn_qkvpacked_func
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel


class StripedMambaConfig(PretrainedConfig):
    model_type = "striped_mamba"

    def __init__(
        self,
        vocab_size: int = 11,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        attention_dropout: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        rms_norm_eps: float = None,  # pytorch RMSNorm default: None
        layer_norm_eps: float = 1e-12,  # pytorch LayerNorm default: 1e-12
        mamba_rmsnorm: bool = True,
        problem_type: Optional[str] = None,
        num_labels: Optional[int] = 2,
        dual_residual: bool = False,  # check
        enc_res_input_norm_scale: float = 0.1,
        enc_alpha: float = 2.0,
        initializer_range: float = 0.02,
        emb_layer_norm_before: bool = True,
        hidden_dropout_prob: float = 0.1,
        position_encoding_type: str = "alibi",  # "alibi" or "rope"
        rope_theta: float = 10000.0,  # for RoPE
        add_pooling_layer: bool = False,  # default: False
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.add_pooling_layer = add_pooling_layer

        # mamba-2 block
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba_rmsnorm = mamba_rmsnorm

        # Normalization
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_eps = layer_norm_eps

        # ResiDual
        self.dual_residual = dual_residual
        self.enc_alpha = enc_alpha
        self.enc_res_input_norm_scale = enc_res_input_norm_scale

        # Embedding
        self.emb_layer_norm_before = emb_layer_norm_before
        self.position_encoding_type = position_encoding_type
        self.rope_theta = rope_theta

        super().__init__(
            num_labels=num_labels,
            problem_type=problem_type,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class PositionalEncoding(nn.Module):
    """
    A module for applying positional encoding to input tensors, supporting both ALiBi and RoPE encoding types.
    Args:
        config (Config): Configuration object containing the following attributes:
            - position_encoding_type (str): Type of positional encoding to use ("alibi" or "rope").
            - num_attention_heads (int): Number of attention heads.
            - hidden_size (int): Size of the hidden layer.
            - rope_theta (float): Theta parameter for RoPE encoding.
    Attributes:
        position_encoding_type (str): Type of positional encoding being used.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        hidden_size (int): Size of the hidden layer.
        alibi_slopes (torch.Tensor): Slopes for ALiBi encoding (if applicable).
        rope_cache (dict): Cache for RoPE frequencies (if applicable).
        theta (float): Theta parameter for RoPE encoding (if applicable).
    Methods:
        _get_alibi_slopes(num_heads):
            Computes the slopes for ALiBi encoding based on the number of attention heads.
        _get_rope_freqs(seq_len, device):
            Computes and caches the RoPE frequencies for a given sequence length and device.
        _rotate_half(x):
            Rotates half of the hidden dimensions of the input tensor.
        apply_rope(qkv):
            Applies RoPE encoding to the query, key, and value tensors.
    """

    def __init__(self, config):
        super().__init__()
        self.position_encoding_type = config.position_encoding_type
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        if self.position_encoding_type == "alibi":
            self.alibi_slopes = self._get_alibi_slopes(self.num_heads)
        elif self.position_encoding_type == "rope":
            self.rope_cache = {}
            self.theta = config.rope_theta
        else:
            raise ValueError(
                f"Unknown position encoding type: {self.position_encoding_type}"
            )

    def _get_alibi_slopes(self, num_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_heads = num_heads - closest_power_of_2
            if extra_heads > 0:
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                    :extra_heads
                ]
                slopes.extend(extra_slopes)

        return torch.tensor(slopes, dtype=torch.float32)

    def _get_rope_freqs(self, seq_len: int, device: torch.device):
        if seq_len not in self.rope_cache:
            theta = self.theta
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim must be even, got {self.head_dim}")
            dim = self.head_dim // 2

            # Create position encodings [seq_len]
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)

            # Create frequency bands [dim]
            freqs = torch.arange(0, dim, device=device, dtype=torch.float32)
            inv_freq = 1.0 / (theta ** (freqs / dim))

            # Compute frequencies [seq_len, dim]
            emb = torch.einsum("n,d->nd", pos, inv_freq)

            # Cache the frequencies
            self.rope_cache[seq_len] = (
                torch.cos(emb),  # [seq_len, dim]
                torch.sin(emb),  # [seq_len, dim]
            )

        return self.rope_cache[seq_len]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, num_heads, head_dim = qkv.shape

        # Get RoPE frequencies
        freqs_cos, freqs_sin = self._get_rope_freqs(
            seq_len, qkv.device
        )  # [seq_len, dim]

        # Prepare cos and sin embeddings for broadcasting
        # [seq_len, dim] -> [1, seq_len, 1, 1, head_dim]
        freqs_cos = freqs_cos.view(1, seq_len, 1, 1, self.head_dim // 2)
        freqs_sin = freqs_sin.view(1, seq_len, 1, 1, self.head_dim // 2)

        # Repeat embeddings for all batches and heads
        # [1, seq_len, 1, 1, head_dim] -> [batch_size, seq_len, 1, num_heads, head_dim]
        freqs_cos = freqs_cos.expand(batch_size, seq_len, 1, num_heads, -1)
        freqs_sin = freqs_sin.expand(batch_size, seq_len, 1, num_heads, -1)

        # Concatenate to match the full head_dim
        # [batch_size, seq_len, 1, num_heads, head_dim]
        freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
        freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)

        # Apply RoPE to Q and K
        for i in range(2):  # Apply only to Q and K
            q = qkv[:, :, i]  # [batch_size, seq_len, num_heads, head_dim]
            q_rotated = self._rotate_half(q)
            qkv[:, :, i] = q * freqs_cos.squeeze(2) + q_rotated * freqs_sin.squeeze(2)

        return qkv


class SwiGLU(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype, bias=False)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype, bias=False)

    def forward(self, x):
        output = self.w1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.w2(x)
        return swiglu


class FlashAttentionBlock(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.position_encoding = PositionalEncoding(config)

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, dtype=dtype, bias=True
        )
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, dtype=dtype, bias=False
        )

    def forward(self, x, attention_mask=None, causal=False):
        batch_size, seq_len, _ = x.shape

        x = x.to(torch.bfloat16)  # for flash attention
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        if self.position_encoding.position_encoding_type == "alibi":
            alibi_slopes = self.position_encoding.alibi_slopes.to(
                device=x.device, dtype=torch.float32
            )
            if len(alibi_slopes.shape) == 1:
                alibi_slopes = alibi_slopes.expand(batch_size, -1)

            output = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attention_dropout,
                causal=causal,
                alibi_slopes=alibi_slopes,
                softmax_scale=None,
                window_size=(-1, -1),
            )
        else:  # rope
            qkv = self.position_encoding.apply_rope(qkv)
            output = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attention_dropout,
                causal=causal,
                alibi_slopes=None,
                softmax_scale=None,
                window_size=(-1, -1),
            )

        output = output.reshape(batch_size, seq_len, self.hidden_size).to(torch.float32)
        output = self.o_proj(output)

        return output


class MambaBlock(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_dual_residual = getattr(config, "dual_residual", True)
        self.residual_scale = getattr(config, "enc_res_input_norm_scale", 1.0)

        self.mamba = Mamba2(
            d_model=config.hidden_size,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dtype=dtype,
            rmsnorm=config.mamba_rmsnorm,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.swiglu = SwiGLU(config, dtype=dtype)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, data, attention_mask=None):
        """
        Args:
            data: Tuple[Tensor, Tensor] containing:
                - x: main path tensor for Post-LN (batch_size, seq_len, hidden_size)
                - x_d: Pre-LN path tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
        Returns:
            Tuple[Tensor, Tensor]:
                - x_a: Post-LN path output after residual connection
                - x_d: Updated Pre-LN path tensor
        """
        x, x_d = data
        if self.use_dual_residual:
            # Post-LN path
            x_ln = self.norm(x)
            x_f = self.mamba(x_ln)
            x_f = self.dropout(x_f)
            x_a = x + x_f

            # Pre-LN path update
            x_f_scaled = x_f * self.residual_scale

            # FFN
            x_ln = self.norm(x_a)
            x_f = self.swiglu(x_ln)
            x_f = self.dropout(x_f)
            x_a = x_a + x_f

            # Pre-LN path update
            x_f_scaled = x_f_scaled + x_f * self.residual_scale

            if x_d is None:
                x_d = x_f_scaled
            else:
                x_d = x_d + x_f_scaled

            return x_a, x_d
        else:
            x_ln = self.norm(x)
            x_f = self.mamba(x_ln)
            x = x + self.dropout(x_f)
            x_f = self.swiglu(self.norm(x))
            x = x + self.dropout(x_f)
            return x, None


class AttentionBlock(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()
        self.flash_attn = FlashAttentionBlock(config, dtype=dtype)
        self.swiglu = SwiGLU(config, dtype=dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.use_dual_residual = getattr(config, "dual_residual", True)
        self.residual_scale = getattr(config, "enc_res_input_norm_scale", 1.0)

    def forward(self, data, attention_mask=None):
        """
        Args:
            data: Tuple[Tensor, Tensor] containing:
                - x: main path tensor for Post-LN (batch_size, seq_len, hidden_size)
                - x_d: Pre-LN path tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
        Returns:
            Tuple[Tensor, Tensor]:
                - x_a: Post-LN path output after residual connection
                - x_d: Updated Pre-LN path tensor
        """
        x, x_d = data
        if self.use_dual_residual:
            # Post-LN path
            x_ln = self.norm(x)
            x_f = self.flash_attn(x_ln, attention_mask)
            x_f = self.dropout(x_f)
            x_a = x + x_f

            # Pre-LN path update
            x_f_scaled = x_f * self.residual_scale

            # FFN
            x_ln = self.norm(x_a)
            x_f = self.swiglu(x_ln)
            x_f = self.dropout(x_f)
            x_a = x_a + x_f

            # Pre-LN path update
            x_f_scaled = x_f_scaled + x_f * self.residual_scale

            if x_d is None:
                x_d = x_f_scaled
            else:
                x_d = x_d + x_f_scaled

            return x_a, x_d
        else:
            x_ln = self.norm(x)
            x_f = self.flash_attn(x_ln, attention_mask)
            x = x + self.dropout(x_f)
            x_f = self.swiglu(self.norm(x))
            x = x + self.dropout(x_f)
            return x, None


class StripedMambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = StripedMambaConfig
    base_model_prefix = "striped_mamba"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MambaBlock", "AttentionBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)


class StripedMambaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.emb_scale = math.sqrt(config.hidden_size)

        if config.emb_layer_norm_before:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds * self.emb_scale

        if self.norm is not None:
            embeddings = self.norm(embeddings)

        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(
                embeddings.dtype
            )

        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(
        self, input_ids, padding_idx, past_key_values_length=0
    ):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class StripedMambaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class StripedMambaModel(StripedMambaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = StripedMambaEmbeddings(config)

        # layers in 75:25 ratio
        self.blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if (i + 1) % 4 == 0:  # 25%: attention
                self.blocks.append(AttentionBlock(config))
            else:  # 75%: mamba
                self.blocks.append(MambaBlock(config))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pooler = StripedMambaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}"""
        for layer, heads in heads_to_prune.items():
            if isinstance(self.blocks[layer], AttentionBlock):
                self.blocks[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get embeddings
        x = self.embeddings(
            input_ids=input_ids,
            # position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # Initialize both paths
        x_d = None  # Pre-LN path will be accumulated through layers

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Process through blocks
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            # Forward through block with both paths
            x, x_d = block((x, x_d), attention_mask)

            if output_attentions and isinstance(block, AttentionBlock):
                all_self_attentions = all_self_attentions + (block.attention_weights,)

        # Final layer norm for both paths
        # x_ln_N+1 = LN(x_N)
        x_ln = self.norm(x)

        # Combine paths for final output
        # y = x_ln_N+1 + LN(x_d_N+1)
        if x_d is not None:
            x_d_ln = self.norm(x_d)
            # hidden_states = x_ln + x_d_ln
            hidden_states = self.config.enc_alpha * x_ln + x_d_ln
        else:
            hidden_states = x_ln

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class StripedMambaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.decoder(x) + self.bias
        return x


class StripedMambaForMaskedLM(StripedMambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.mamba = StripedMambaModel(
            config, add_pooling_layer=config.add_pooling_layer
        )
        self.lm_head = StripedMambaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.mamba.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.mamba.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mamba(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class StripedMambaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class StripedMambaForSequenceClassification(StripedMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Initialize Mamba model and classification head
        self.mamba = StripedMambaModel(
            config, add_pooling_layer=config.add_pooling_layer
        )
        self.classifier = StripedMambaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get model outputs
        outputs = self.mamba(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class StripedMambaForTokenClassification(StripedMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mamba = StripedMambaModel(
            config, add_pooling_layer=config.add_pooling_layer
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mamba(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
