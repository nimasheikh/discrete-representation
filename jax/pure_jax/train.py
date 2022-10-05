import jax.numpy as jnp
from jax import grad, jit, vmap, random
import jax
import optax
import numpy as np


def accuracy(params, images, targets, predict_func):
    predicted_class = jnp.argmax(predict_func(params, images), axis=1).squeeze()
    return jnp.mean(predicted_class == targets.squeeze())

def loss(params, images, targets, predict_func):
    preds = predict_func(params, images)
    return -jnp.mean(preds * targets)

def loss_optax(params, images, targets, predict_func,
    rkey, noise_status='uniform'):
    preds = predict_func(params, images, rkey, noise_status=noise_status)
    return optax.softmax_cross_entropy_with_integer_labels(preds, targets).mean()

@jit
def update(params, x, y, step_size=1e-2):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)]

def fit(params: optax.Params, optimizer: optax.GradientTransformation,
    DATAGEN, keygen, loss_func, noise_status='uniform') -> optax.Params:
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, batch, labels, rkey):
        loss_value, grads = jax.value_and_grad(loss_func)(params, batch,\
             labels, rkey)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i, (batch, labels) in enumerate(DATAGEN):
        params, opt_state, loss_value = step(params, opt_state, batch, labels, keygen())
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')

    return params
