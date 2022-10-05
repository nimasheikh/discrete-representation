import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
# a simple MLP model

def mlp_random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def mlp_init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [mlp_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def mlop_predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)