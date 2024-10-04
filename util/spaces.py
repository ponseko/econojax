import chex
import jax.numpy as jnp
import jax
import numpy as np


class MultiDiscrete(object):
    """
    Minimal implementation of a MultiDiscrete space.
    input nvec: array of integers representing the number of discrete values in each dimension
    """

    def __init__(self, nvec: chex.Array, dtype: jnp.dtype = jnp.int8, start: int = 0):
        assert (
            len(nvec.shape) == 1 and nvec.shape[0] > 0
        ), "nvec must be a 1D array with at least one element"
        assert np.all(nvec > 0), "All elements in nvec must be greater than 0"
        self.nvec = nvec
        self.shape = nvec.shape
        self.n = self.shape[0]
        self.num_action_types = self.shape[0]
        self.num_actions_per_type = nvec
        self.dtype = dtype
        self.start = start

        # check if all elements in nvec are the same
        if np.all(nvec == nvec[0]):
            self.uniform = True
        else:
            self.uniform = False
            raise NotImplementedError(
                "Non-uniform MultiDiscrete spaces are not supported"
            )

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, self.shape, self.start, self.nvec)

    def contains(self, x: chex.Array) -> bool:
        return (
            x.shape == self.shape
            and jnp.all(x >= self.start)
            and jnp.all(x < self.nvec)
        )
    

