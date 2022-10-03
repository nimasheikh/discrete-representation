from jax import numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random
from jax.scipy.special import logsumexp
from flax.training import train_state
import optax
import jax
from models import CNN, simpleMLP


model_class = simpleMLP

def model_with_vec(vec, batch, value_tree):
  """
  A wrapper around the model which gets the parameters
  in shape of a flat vector and turns them into a value_tree
  and applies the model. 


  This is usefulee since the gradients and hessian will be in
  natural matrix form. 
  """
  
  pass




def one_hot(x, num_classes=10, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(num_classes), dtype)

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def cross_entropy_loss(*,logits, labels):
    labels_onehot = one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = model_class()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)

        
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = model_class().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics


def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state

  
@jax.jit
def eval_step(params, batch):
  logits = model_class().apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])

def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']