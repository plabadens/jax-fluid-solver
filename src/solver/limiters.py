import chex
import jax.numpy as jnp

from jax import Array, jit
from functools import partial


@partial(jit, static_argnums=(1))
def left_slope(function_values: Array, axis: int) -> Array:
    return function_values - jnp.roll(function_values, 1, axis)


@partial(jit, static_argnums=(1))
def monotonized_central(function_values: Array, n: int) -> Array:
    """Monotonized central slope limiter

    Args:
        function_values: Array of shape (n, n) containing the function values
        n: Number of cells in the grid

    Returns:
        Array of shape (2, n, n) containing the slopes in the x and y directions
    """
    slopes = []

    for i in range(2):
        left = left_slope(function_values, axis=i)
        right = jnp.roll(left, -1, axis=i)
        slopes_dim = jnp.where(
            left * right > 0, 2.0 * left * right / (left + right), 0.0
        )
        slopes.append(slopes_dim)

    slopes = jnp.stack(slopes)

    return slopes
