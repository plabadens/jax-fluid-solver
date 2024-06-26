import chex
import jax.debug as jdb
import jax.numpy as jnp
import solver.hydro as hydro

from jax import Array, jit
from solver.hydro import PrimitiveVariable, ConservativeVariable


@jit
def local_lax_friedrichs(
    left: PrimitiveVariable, right: PrimitiveVariable, adiabatic_index: float
) -> ConservativeVariable:
    """
    Local Lax-Friedrichs Riemann solver.

    Args:
        left (PrimitiveVariable): The left state.
        right (PrimitiveVariable): The right state.
        adiabatic_index (float): The adiabatic index (gamma).

    Returns:
        ConservativeVariable: The flux computed using the Local Lax-Friedrichs Riemann solver.
    """
    cs_left = jnp.nan_to_num(jnp.sqrt(adiabatic_index * left.pressure / left.density))
    cs_right = jnp.nan_to_num(
        jnp.sqrt(adiabatic_index * right.pressure / right.density)
    )
    cs_max = jnp.maximum(cs_left, cs_right)

    propagation_max = jnp.maximum(
        jnp.abs(left.velocity_x) + cs_max, jnp.abs(right.velocity_x) + cs_max
    )

    cons_left = hydro.primitive_to_conservative(left, adiabatic_index=adiabatic_index)
    cons_right = hydro.primitive_to_conservative(right, adiabatic_index=adiabatic_index)

    flux_left = hydro.hydrodynamic_flux(left, cons_left)
    flux_right = hydro.hydrodynamic_flux(right, cons_right)

    flux_density = 0.5 * (
        flux_left.density
        + flux_right.density
        - propagation_max * (cons_right.density - cons_left.density)
    )

    flux_momentum_x = 0.5 * (
        flux_left.momentum_x
        + flux_right.momentum_x
        - propagation_max * (cons_right.momentum_x - cons_left.momentum_x)
    )

    flux_momentum_y = 0.5 * (
        flux_left.momentum_y
        + flux_right.momentum_y
        - propagation_max * (cons_right.momentum_y - cons_left.momentum_y)
    )

    flux_total_energy = 0.5 * (
        flux_left.total_energy
        + flux_right.total_energy
        - propagation_max * (cons_right.total_energy - cons_left.total_energy)
    )

    if left.dye_density is not None and right.dye_density is not None:
        flux_dye_density = 0.5 * (
            flux_left.dye_density
            + flux_right.dye_density
            - propagation_max * (cons_right.dye_density - cons_left.dye_density)
        )
    else:
        flux_dye_density = None

    return ConservativeVariable(
        density=flux_density,
        dye_density=flux_dye_density,
        momentum_x=flux_momentum_x,
        momentum_y=flux_momentum_y,
        total_energy=flux_total_energy,
    )
