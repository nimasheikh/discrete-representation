from jax import random

class keyGen():
    def __init__(self, key=random.PRNGKey(0)):
        _, self.subkey = random.split(key)

    def __call__(self):
        temp, self.subkey = random.split(self.subkey)
        return temp

