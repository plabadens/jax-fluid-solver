import jax.numpy as jnp

from solver.solvers import local_lax_friedrichs, PrimitiveVariable


def test_local_lax_friedrichs():
    left = PrimitiveVariable(
        density=jnp.ones((10, 10)),
        velocity_x=jnp.ones((10, 10)),
        velocity_y=jnp.ones((10, 10)),
        pressure=jnp.ones((10, 10)),
    )
    right = PrimitiveVariable(
        density=jnp.ones((10, 10)) * 2,
        velocity_x=jnp.ones((10, 10)) * 2,
        velocity_y=jnp.ones((10, 10)) * 2,
        pressure=jnp.ones((10, 10)) * 2,
    )
    adiabatic_index = 1.4

    result = local_lax_friedrichs(left, right, adiabatic_index)

    assert result.density.shape == (10, 10)
    assert result.momentum_x.shape == (10, 10)
    assert result.momentum_y.shape == (10, 10)
    assert result.total_energy.shape == (10, 10)
