from typing import Dict, Generator, Optional, Tuple
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.sample_utils import top_p_sampling

from ..shard import Shard

# TODO: support a speculative model so we can parallelise compute across devices
class StatefulShardedModel:
  def __init__(self, shard: Shard, model: nn.Module, max_kv_size: int = 1024, max_caches: int = 2):
    self.shard = shard
    self.model = model
    self.max_kv_size = max_kv_size
    self.max_caches = max_caches
    self.caches = OrderedDict()

  def step(
    self,
    request_id: str,
    input_ids,
    pixel_values=None,
    aspect_ratio_ids=None,
    aspect_ratio_mask=None,
    inference_state: Optional[mx.array] = None,
    temp: float = 0.0,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
  ) -> Generator[Tuple[mx.array, mx.array], None, None]:
    def sample(logits: mx.array) -> Tuple[mx.array, float]:
      if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))
        logits[:, indices] += values

      if temp == 0:
        token = mx.argmax(logits, axis=-1)
      else:
        if top_p > 0 and top_p < 1.0:
          token = top_p_sampling(logits, top_p, temp)
        else:
          token = mx.random.categorical(logits*(1/temp))

      return token

    y = input_ids

    if request_id not in self.caches:
      self.init_cache(request_id)
    else:
      self.caches.move_to_end(request_id)

    cache = self.caches[request_id]

    if pixel_values is None:
      if self.shard.is_first_layer() and y.ndim==1:
        y = y[None]
      output, inference_state = self.model(y, cache=cache, inference_state=inference_state)
    else:
      output, inference_state = self.model(y, pixel_values=pixel_values, aspect_ratio_ids=aspect_ratio_ids, aspect_ratio_mask=aspect_ratio_mask, cache=cache, inference_state=inference_state)

    if self.shard.is_last_layer():
      logits = output[:, -1, :]
      y = sample(logits)
      return y, inference_state
    else:
      return output, inference_state

  def __call__(
    self,
    request_id: str,
    input_ids,
    pixel_values=None,
    aspect_ratio_ids=None,
    aspect_ratio_mask=None,
    temp: float = 0.0,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
  ) -> Generator[Tuple[mx.array, mx.array], None, None]:
    return self.step(request_id, input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, temp=temp, top_p=top_p, logit_bias=logit_bias)

  def init_cache(self, request_id: str):
    kv_heads = ([self.model.n_kv_heads]*len(self.model.layers) if isinstance(self.model.n_kv_heads, int) else self.model.n_kv_heads)
    if self.max_kv_size is not None:
      cache = [RotatingKVCache(self.model.head_dim, n, max_size=self.max_kv_size, keep=4) for n in kv_heads]
    else:
      cache = [KVCache(self.model.head_dim, n) for n in kv_heads]

    if len(self.caches) >= self.max_caches:
      self.caches.popitem(last=False)

    self.caches[request_id] = cache
