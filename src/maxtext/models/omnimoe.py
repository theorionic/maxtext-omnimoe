# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmniMoE model definition.

OmniMoE combines:
  - DeepSeek-V4 MoE routing: sqrt(softplus) affinity, noaux_tc bias tempering,
    SwiGLU clamping (limit=10), fine-grained experts + 1 shared expert.
  - Qwen3.6 attention geometry: 8:1 GQA, partial RoPE, QK-RMSNorm.
  - Hybrid attention (our own design): every Nth layer full causal (GLOBAL),
    rest sliding-window (LOCAL_SLIDING). Distinct from Qwen3.6 DeltaNet hybrid
    and DeepSeek-V4 CSA/HCA.
  - Standard residuals (no mHC).
  - Standard SDPA attention (no MLA, no compressed attention).

This layer reuses MaxText's existing RoutedAndSharedMoE (which already supports
sqrtsoftplus, routed_bias, mlp_activations_limit) and standard Attention (which
already supports GQA, partial RoPE, QK-norm, sliding window). The only new
logic is the per-layer attention_type dispatch for the hybrid pattern.
"""

from typing import Optional

from flax import nnx
import flax.linen as nn
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import Dropout
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical


def get_attention_type(layer_idx: int, full_every_n: int) -> AttentionType:
  if full_every_n <= 1:
    return AttentionType.GLOBAL
  return AttentionType.GLOBAL if (layer_idx % full_every_n == 0) else AttentionType.LOCAL_SLIDING


class OmniMoEDecoderLayer(nnx.Module):

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.layer_idx = layer_idx

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    full_every_n = getattr(config, "omnimoe_full_causal_every_n", 4)
    attn_type = get_attention_type(layer_idx, full_every_n)

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )
    self.post_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        attention_type=attn_type,
        sliding_window_size=config.sliding_window_size,
        use_qk_norm=config.use_qk_norm,
        partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        query_pre_attn_scalar=config.head_dim ** -0.5,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    kernel_init = initializers.nd_dense_init(
        config.dense_init_scale, "fan_in", "truncated_normal")
    if config.shared_experts > 0:
      self.mlp = moe.RoutedAndSharedMoE(
          config=config,
          mesh=mesh,
          kernel_init=kernel_init,
          kernel_axes=("embed", None),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          is_hash_routing=False,
          rngs=rngs,
      )
    else:
      # Pure routed MoE (no always-on shared expert): each token is dispatched to
      # exactly its top-k routed experts and nothing else. Used for the
      # one-expert-per-chip expert-parallel configuration.
      self.mlp = moe.RoutedMoE(
          config=config,
          num_experts=config.num_experts,
          num_experts_per_tok=config.num_experts_per_tok,
          mesh=mesh,
          kernel_init=kernel_init,
          kernel_axes=("embed_moe", None),
          intermediate_dim=config.moe_mlp_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          is_hash_routing=False,
          rngs=rngs,
      )

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = (
          "activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = (
          "activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_axis_names = (
        "activation_batch", "activation_norm_length", "activation_mlp")

  def _shard(self, x, mlp=False):
    return maybe_shard_with_logical(
        x, logical_axes=self.mlp_axis_names if mlp else self.activation_axis_names,
        mesh=self.mesh, shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1, rules=self.config.logical_axis_rules)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    x = self._shard(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    lnx = self._shard(self.pre_self_attention_norm(x))
    attn_out, kv_cache = self.self_attention(
        lnx, lnx, decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attn_out = self._shard(attn_out)
    intermediate = x + attn_out
    hidden = self._shard(self.post_self_attention_norm(intermediate))

    mlp_out, load_balance_loss, moe_bias_updates = self.mlp(
        inputs=hidden, input_ids=decoder_input_tokens)
    mlp_out = self._shard(mlp_out, mlp=True)

    layer_output = intermediate + mlp_out
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = self._shard(layer_output)

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache


OmniMoEDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    OmniMoEDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)