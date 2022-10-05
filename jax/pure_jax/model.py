import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
from jax import nn
# a simple MLP model

def mlp_random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def mlp_init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [mlp_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def mlp_representation(params, image):
    activations = image
    for w, b in params[:-2]:
        outputs = jnp.dot(activations, w) + b
        activations = nn.relu(outputs)

    final_w, final_b = params[-2]
    representation = nn.sigmoid(jnp.dot(activations, final_w) + final_b)
    return representation

def mlp_predict(params, image, key, noise_status='uniform'):
    representations = mlp_representation(params, image) 
    if noise_status == 'uniform':
        noise = random.uniform(key, representations.shape, minval=0,maxval=0.5)
    else:
        noise = jnp.zeros(representations.shape)

    final_w, final_b = params[-1]
    logits = jnp.dot(representations + noise,  final_w) + final_b
    return logits


