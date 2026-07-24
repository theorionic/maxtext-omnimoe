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

"""Periodic in-training text generation for the pre-train loop.

Runs a small, fixed-shape greedy decode every ``generate_interval`` optimizer
steps and logs the sampled continuations. Deliberately simple:

  * No KV cache. Each new token is produced by a full forward over a fixed
    ``[B, prompt_len + i]``-padded ``[B, max_len]`` buffer, so it reuses the
    training model's forward + sharding exactly and adds no decode/prefill
    machinery. The forward is jitted once (constant shape); the greedy argmax
    loop runs in Python (max_new_tokens iterations, tiny compared to a step).
  * Works for both Flax NNX models (e.g. OmniMoE) and Linen models.
  * Never touches the train step and is wrapped so any failure is logged and
    training continues. Fully opt-in via ``config.generate_interval > 0``.

This is a qualitative health probe (is the model producing more coherent text
over time?), not an efficient serving path.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from maxtext.input_pipeline.tokenizer import build_tokenizer
from maxtext.utils import max_logging

# Cache tokenizer + jitted forward across generate_interval calls.
_CACHE: dict = {"tokenizer": None, "linen_fn": None, "nnx_fn": None}


def _get_tokenizer(config):
  if _CACHE["tokenizer"] is None:
    _CACHE["tokenizer"] = build_tokenizer(
        config.tokenizer_path,
        config.tokenizer_type,
        add_bos=getattr(config, "add_bos", True),
        add_eos=False,
        hf_access_token=getattr(config, "hf_access_token", ""),
    )
  return _CACHE["tokenizer"]


def _prompts(config) -> list[str]:
  raw = getattr(config, "generate_prompts", "") or ""
  prompts = [p for p in raw.split("|||") if p.strip()]
  return prompts or ["The meaning of life is"]


def _linen_forward(model, params, tokens, positions, segs):
  logits, _ = model.apply(
      params,
      tokens,
      positions,
      decoder_segment_ids=segs,
      enable_dropout=False,
      rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(0)},
      mutable=["intermediates"],
      decoder_target_tokens=tokens,
      decoder_target_mask=segs,
  )
  return logits


@functools.partial(jax.jit, static_argnums=(0,))
def _nnx_forward(graphdef, gen_state, tokens, positions, segs):
  # Functional NNX call: rebuild the module from (static graphdef, immutable state)
  # inside jit so nothing on the live training state is mutated, donated, or freed.
  # Only logits are returned; any intermediates sown during the forward are DCE'd.
  model = nnx.merge(graphdef, gen_state)
  logits = model(
      decoder_input_tokens=tokens,
      decoder_positions=positions,
      decoder_segment_ids=segs,
      enable_dropout=False,
      decoder_target_tokens=tokens,
      decoder_target_mask=segs,
  )
  return logits


def _greedy(forward, tokens, positions, prompt_len, max_len, eos):
  """Python greedy loop calling a jitted single-forward `forward(tokens, positions, segs)`."""
  for i in range(prompt_len, max_len):
    segs = (np.arange(max_len)[None, :] < i).astype(np.int32)
    logits = forward(jnp.asarray(tokens), jnp.asarray(positions), jnp.asarray(segs))
    next_tok = np.asarray(jax.device_get(jnp.argmax(logits[:, i - 1, :], axis=-1)))
    tokens[:, i] = next_tok
  return tokens


def generate_and_log(model, state, config, mesh, logical_axis_rules, step: int) -> None:
  """Greedy-decode a few prompts with the current params and log the text."""
  try:
    tokenizer = _get_tokenizer(config)
    prompts = _prompts(config)
    max_new = int(getattr(config, "generate_max_new_tokens", 48))

    # Pad the batch up to the device count so activations shard evenly over the
    # (data, fsdp, expert) axes like training does.
    n_dev = jax.device_count()
    batch = max(n_dev, ((len(prompts) + n_dev - 1) // n_dev) * n_dev)

    enc = [tokenizer.encode(p) for p in prompts]
    prompt_len = max(len(e) for e in enc)
    max_len = prompt_len + max_new

    tokens = np.zeros((batch, max_len), dtype=np.int32)
    for r, ids in enumerate(enc):
      tokens[r, : min(len(ids), prompt_len)] = ids[:prompt_len]
    positions = np.broadcast_to(np.arange(max_len, dtype=np.int32), (batch, max_len)).copy()

    is_linen = isinstance(model, nn.Module)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(logical_axis_rules):
      if is_linen:
        if _CACHE["linen_fn"] is None:
          _CACHE["linen_fn"] = jax.jit(functools.partial(_linen_forward, model))
        fwd = functools.partial(_CACHE["linen_fn"], state.params)
      else:
        # `model` is the live NNX module (structure only is needed); the CURRENT
        # params live in `state` (the TrainStateNNX state), with the model's params
        # under the 'model' substate. Merge the module graphdef with that substate
        # inside jit so nothing on the live training state is mutated or donated.
        graphdef = nnx.graphdef(model)
        try:
          gen_state = state["model"]
        except Exception:  # pylint: disable=broad-except
          gen_state = getattr(state, "model", state)
        fwd = functools.partial(_nnx_forward, graphdef, gen_state)
      tokens = _greedy(fwd, tokens, positions, prompt_len, max_len, tokenizer.eos_id)
  except Exception as e:  # pylint: disable=broad-except
    # Generation is a best-effort probe; never let it interrupt training.
    max_logging.log(f"[generate] step {step}: generation skipped ({type(e).__name__}: {e}).")
    return

  eos = tokenizer.eos_id
  max_logging.log(f"===== sample generations @ step {step} =====")
  for r, p in enumerate(prompts):
    gen_ids = tokens[r, len(enc[r]) : max_len].tolist()
    if eos is not None and eos in gen_ids:
      gen_ids = gen_ids[: gen_ids.index(eos)]
    try:
      cont = tokenizer.decode([int(t) for t in gen_ids])
    except Exception:  # pylint: disable=broad-except
      cont = f"<detokenize failed for ids {gen_ids[:8]}...>"
    max_logging.log(f"[gen {r}] prompt={p!r}\n        -> {cont!r}")
  max_logging.log("=============================================")
