import chex
import jax.numpy as jnp
import solver.hydro as hydro
import solver.limiters as limiters
import solver.solvers as solvers

from jax import Array, jit
from typing import Callable
from functools import partial
from solver.hydro import HydroState


@chex.chexify
# @partial(jit, static_argnums=(1, 2, 3))
def muscl_2d(
    state: HydroState,
    dt: float,
    slope_limiter: Callable = limiters.monotonized_central,
    riemann_solver: Callable = solvers.local_lax_friedrichs,
) -> HydroState:
    """MUSCL method for 2D hydrodynamics

    Args:
        state: HydroState dataclass
        dt: Time step
        slope_limiter: Slope limiter function
        riemann_solver: Riemann solver function

    Returns:
        Updated HydroState dataclass
    """
    ds = state.ds
    dt_ds = dt / ds

    # primitive variables
    density = state.density
    momentum_x = state.momentum_x
    momentum_y = state.momentum_y
    total_energy = state.total_energy

    chex.assert_equal_shape(
        [density, momentum_x, momentum_y, total_energy],
        custom_message="Initial state shape mismatch",
    )
    chex.assert_shape(density, (state.n, state.n))

    velocity = hydro.velocity(state)
    chex.assert_shape(velocity, (2, state.n, state.n))

    velocity_x = velocity[0]
    velocity_y = velocity[1]
    pressure = hydro.pressure(state)

    chex.assert_equal_shape(
        [velocity_x, velocity_y, pressure],
        custom_message="Initial state shape mismatch",
    )

    # slopes
    d_density = slope_limiter(density, state.n) / ds
    d_pressure = slope_limiter(pressure, state.n) / ds
    d_velocity_x = slope_limiter(velocity_x, state.n) / ds
    d_velocity_y = slope_limiter(velocity_y, state.n) / ds
    d_velocity = jnp.stack([d_velocity_x, d_velocity_y])

    # trace forward
    divergence = d_velocity[0, 0] + d_velocity[1, 1]
    density_t = (
        -velocity_x * d_density[0] - velocity_y * d_density[1] - density * divergence
    )
    pressure_t = (
        -velocity_x * d_pressure[0]
        - velocity_y * d_pressure[1]
        - pressure * divergence * state.adiabatic_index
    )
    velocity_t = jnp.empty_like(velocity)
    velocity_t = velocity_t.at[0].set(
        -velocity_x * d_velocity_x[0]
        - velocity_y * d_velocity_x[1]
        - d_pressure[0] / density
    )
    velocity_t = velocity_t.at[1].set(
        -velocity_x * d_velocity_y[0]
        - velocity_y * d_velocity_y[1]
        - d_pressure[1] / density
    )

    chex.assert_shape(density, (state.n, state.n))

    # trace back
    for axis in range(2):
        adiabatic_index = state.adiabatic_index

        # Calculates the parallel and perpendicular velocities to the coordinate direction
        # X-axis: iu=0, iv=1, Y-axis: iu=1, iv=0
        iu = axis
        iv = (axis + 1) % 2
        # Parallel (U) and perpendicular (V) velocities
        U = velocity[iu]
        dU = d_velocity[iu]
        U_t = velocity_t[iu]

        V = velocity[iv]
        dV = d_velocity[iv]
        V_t = velocity_t[iv]

        # left state
        density_left = density + 0.5 * (dt * density_t + ds * d_density[axis])
        pressure_left = pressure + 0.5 * (dt * pressure_t + ds * d_pressure[axis])
        U_left = U + 0.5 * (dt * U_t + ds * dU[axis])
        V_left = V + 0.5 * (dt * V_t + ds * dV[axis])

        state_left = hydro.PrimitiveVariable(
            density=density_left,
            velocity_x=U_left,
            velocity_y=V_left,
            pressure=pressure_left,
        )

        # right state
        density_right = jnp.roll(density_left, shift=-1, axis=axis)
        pressure_right = jnp.roll(pressure_left, shift=-1, axis=axis)
        U_right = jnp.roll(U_left, shift=-1, axis=axis)
        V_right = jnp.roll(V_left, shift=-1, axis=axis)

        state_right = hydro.PrimitiveVariable(
            density=density_right,
            velocity_x=U_right,
            velocity_y=V_right,
            pressure=pressure_right,
        )

        chex.assert_shape(density, (state.n, state.n), custom_message=f"axis={axis}")

        # Riemann solver
        flux = riemann_solver(state_left, state_right, adiabatic_index)
        chex.assert_shape(
            flux.density, (state.n, state.n), custom_message=f"axis={axis}"
        )

        # update
        density -= dt_ds * (flux.density - jnp.roll(flux.density, shift=1, axis=axis))
        total_energy -= dt_ds * (
            flux.total_energy - jnp.roll(flux.total_energy, shift=1, axis=axis)
        )

        if axis == 0:
            momentum_x -= dt_ds * (
                flux.momentum_x - jnp.roll(flux.momentum_x, shift=1, axis=axis)
            )
            momentum_y -= dt_ds * (
                flux.momentum_y - jnp.roll(flux.momentum_y, shift=1, axis=axis)
            )
        else:
            momentum_x -= dt_ds * (
                flux.momentum_y - jnp.roll(flux.momentum_y, shift=1, axis=axis)
            )
            momentum_y -= dt_ds * (
                flux.momentum_x - jnp.roll(flux.momentum_x, shift=1, axis=axis)
            )

    chex.assert_shape(density, (state.n, state.n))

    chex.assert_equal_shape(
        [state.density, state.momentum_x, state.momentum_y, state.total_energy]
    )

    return HydroState(
        n=state.n,
        ds=state.ds,
        dx=state.dx,
        dy=state.dy,
        x=state.x,
        y=state.y,
        density=density,
        momentum_x=momentum_x,
        momentum_y=momentum_y,
        total_energy=total_energy,
        t=state.t + dt,
        it=state.it + 1,
    )
