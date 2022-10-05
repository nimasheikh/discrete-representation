import jax
import jax.numpy as jnp
from flax import linen as nn

import numpy as np
import optax

class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


class simpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.squeeze().reshape(x.shape[0], -1)
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=50)(x)
        x = nn.Dense(features=10)(x)
        return x
        