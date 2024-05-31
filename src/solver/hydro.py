import jax.numpy as jnp

from jax import Array, jit
from chex import dataclass
from typing import Callable, Optional


@dataclass
class HydroState:
    n: int
    ds: float
    dx: float
    dy: float
    x: Array
    y: Array
    density: Array
    total_energy: Array
    momentum_x: Array
    momentum_y: Array
    t: float = 0.0
    it: int = 0
    adiabatic_index: float = 1.4
    sound_speed: float = jnp.nan
    dye_concentration: Optional[Array] = None


def create_hydro_state(
    n: int, Lbox: float, func: Optional[Callable] = None, adiabatic_index: float = 1.4
) -> HydroState:
    x = jnp.linspace(-Lbox * 0.5, Lbox * 0.5, num=n, endpoint=False) + 0.5 * Lbox / n
    y = x
    ds = Lbox / n

    total_energy = func(x, y, n, ds) if func is not None else jnp.ones((n, n))

    return HydroState(
        n=n,
        ds=ds,
        dx=ds,
        dy=ds,
        x=x,
        y=y,
        density=jnp.ones((n, n)),
        total_energy=total_energy,
        momentum_x=jnp.ones((n, n)),
        momentum_y=jnp.ones((n, n)),
        adiabatic_index=adiabatic_index,
    )


@jit
def velocity(state: HydroState) -> Array:
    v = jnp.stack(
        [state.momentum_x / state.density, state.momentum_y / state.density], axis=0
    )
    return v


@jit
def pressure(state: HydroState) -> Array:
    internal_energy = jnp.abs(
        state.total_energy
        - 0.5 * (state.momentum_x**2 + state.momentum_y**2) / state.density
    )
    return (state.adiabatic_index - 1.0) * internal_energy


@jit
def temperature(state: HydroState) -> Array:
    return pressure(state) / state.density


def courant_condition(state: HydroState, courant_number: float) -> Array:
    v = velocity(state)
    velocity_magnitude = jnp.sqrt(v[0] ** 2 + v[1] ** 2)
    sound_speed = jnp.sqrt(state.adiabatic_index * pressure(state) / state.density)

    return courant_number * state.ds / jnp.max(velocity_magnitude + sound_speed)


@dataclass
class PrimitiveVariable:
    density: Array
    velocity_x: Array
    velocity_y: Array
    pressure: Array
    dye_density: Optional[Array] = None


@dataclass
class ConservativeVariable:
    density: Array
    momentum_x: Array
    momentum_y: Array
    total_energy: Array
    dye_density: Optional[Array] = None


@jit
def primitive_to_conservative(
    variable: PrimitiveVariable, adiabatic_index: float = jnp.nan
) -> ConservativeVariable:
    total_energy = variable.pressure / (
        adiabatic_index - 1.0
    ) + 0.5 * variable.density * (variable.velocity_x**2 + variable.velocity_y**2)

    return ConservativeVariable(
        density=variable.density,
        dye_density=variable.dye_density,
        momentum_x=variable.density * variable.velocity_x,
        momentum_y=variable.density * variable.velocity_y,
        total_energy=total_energy,
    )


@jit
def hydrodynamic_flux(
    primitive: PrimitiveVariable,
    conservative: ConservativeVariable,
    adiabatic_index: float = jnp.nan,
) -> ConservativeVariable:
    return ConservativeVariable(
        density=conservative.momentum_x,
        dye_density=conservative.momentum_x,
        momentum_x=conservative.momentum_x * primitive.velocity_x + primitive.pressure,
        momentum_y=conservative.momentum_y * primitive.velocity_x,
        total_energy=(
            (conservative.total_energy + primitive.pressure) * primitive.velocity_x
        ),
    )
