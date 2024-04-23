import jax
import jax.numpy as jnp
import pytest

from solver.hydro import (
    HydroState,
    PrimitiveVariable,
    ConservativeVariable,
    hydrodynamic_flux,
    primitive_to_conservative,
)


def test_hydrodynamic_flux():
    primitive = PrimitiveVariable(
        density=jnp.ones((10, 10)),
        velocity_x=jnp.ones((10, 10)),
        velocity_y=jnp.ones((10, 10)),
        pressure=jnp.ones((10, 10)),
    )
    conservative = ConservativeVariable(
        density=jnp.ones((10, 10)),
        momentum_x=jnp.ones((10, 10)),
        momentum_y=jnp.ones((10, 10)),
        total_energy=jnp.ones((10, 10)),
    )
    adiabatic_index = 1.4

    result = hydrodynamic_flux(primitive, conservative, adiabatic_index)

    assert result.density.shape == (10, 10)
    assert result.momentum_x.shape == (10, 10)
    assert result.momentum_y.shape == (10, 10)
    assert result.total_energy.shape == (10, 10)


def test_primitive_to_conservative():
    variable = PrimitiveVariable(
        density=jnp.ones((10, 10)),
        velocity_x=jnp.ones((10, 10)),
        velocity_y=jnp.ones((10, 10)),
        pressure=jnp.ones((10, 10)),
    )
    adiabatic_index = 1.4

    result = primitive_to_conservative(variable, adiabatic_index)

    assert result.density.shape == (10, 10)
    assert result.momentum_x.shape == (10, 10)
    assert result.momentum_y.shape == (10, 10)
    assert result.total_energy.shape == (10, 10)
