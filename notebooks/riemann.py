import chex
import jax.numpy as jnp
import jax.lax as lax
import numpy as onp

from jax import Array, jit, tree_util
from jax.typing import ArrayLike
from typing import Optional


# Compute a "left slope". Returns the slope at the cell interface between x_i-1/2 and x_i+1/2
# at the index position i. Left slope is useful for computing slopes
@jit
def left_slope(function_values: ArrayLike, axis=0):
    return function_values - jnp.roll(function_values, 1, axis)


@jit
def monotonized_central(function_values) -> tuple[Array]:
    """Monotonized central slope limiter"""

    if function_values.ndim == 1:
        ls = left_slope()
        rs = jnp.roll(ls, -1)
        slopes = jnp.zeros(ls.shape)
        w = jnp.where(ls * rs > 0.0)
        slopes[w] = 2.0 * ls[w] * rs[w] / (ls[w] + rs[w])

        return (slopes,)
    else:
        shape = jnp.insert(jnp.array(function_values.shape), 0, function_values.ndim)
        slopes = jnp.zeros(shape)

        for i in range(function_values.ndim):
            ls = left_slope(function_values, axis=i)
            rs = jnp.roll(ls, -1, axis=i)
            cs = jnp.zeros(function_values.shape)
            w = jnp.where(ls * rs > 0.0)
            cs[w] = 2.0 * ls[w] * rs[w] / (ls[w] + rs[w])
            slopes[i] = cs

        return (slopes,)


# Define an hydro state including all auxiliary scalars and coordinates in a class
class HydroState(object):
    def __init__(self, n=64, adiabatic_index=1.4, sound_speed=1.0, Lbox=2.0):
        self.n = n
        self.Lbox = Lbox
        self.time = 0.0
        self.iterations = 0
        self.adiabatic_index = adiabatic_index
        if adiabatic_index == 1.0:
            self.sound_speed = sound_speed
        self.density = jnp.ones((n, n))
        self.total_energy = jnp.ones((n, n))
        self.Px = jnp.zeros((n, n))
        self.Py = jnp.zeros((n, n))
        self._initialize_coordinates()

    def _initialize_coordinates(self):
        n = self.n

        self.ds = self.Lbox / n
        self.dx = self.ds
        self.dy = self.ds

        # cell-centered coordinates
        self.x_coords = (
            jnp.linspace(-self.Lbox * 0.5, self.Lbox * 0.5, num=self.n, endpoint=False)
            + 0.5 * self.ds
        )
        self.y_coords = self.x_coords

        self.radius = jnp.sqrt(self.y_coords**2 + self.x_coords**2)

    @property
    def velocity(self):
        """Compute velocity from conservative variables"""
        return jnp.array([self.Px / self.density, self.Py / self.density])

    @property
    def pressure(self):
        """Compute pressure from conservative variables"""
        if self.adiabatic_index == 1.0:
            pressure = self.sound_speed**2 * self.density
        else:
            internal_energy = self.total_energy - 0.5 * self.Px**2 / self.density
            pressure = (self.adiabatic_index - 1.0) * internal_energy
        return pressure

    @property
    def temperature(self):
        """Compute the temperature, defined as  P/rho = kB / mu T"""
        return self.pressure / self.density

    # Courant condition with default Courant number=0.2 for a fluid
    # maximum propagation velocity is max(|v| + sound speed), where max is taken over all cells
    def courant_condition(self, courant_number=0.2):
        """Courant condition for HD"""
        v_x, v_y = self.velocity
        speed = jnp.sqrt(v_x**2 + v_y**2)
        sound_speed = jnp.sqrt(self.adiabatic_index * self.pressure / self.density)
        return courant_number * self.ds / jnp.max(sound_speed + speed)


@chex.dataclass
class Primitive:
    D: Array
    U: Array
    V: Array
    P: Array
    gamma: float = 1.0


@chex.dataclass
class Conservative:
    D: Array
    mU: Array
    mV: Array
    Etot: Optional[Array]


@jit
def primitive_to_conservative(q: Primitive) -> Conservative:
    """Conservative variable from primitive variable"""
    total_energy = (
        q.P / (q.gamma - 1) + 0.5 * q.D * (q.U**2 + q.V**2)
        if q.gamma != 1
        else None
    )

    return Conservative(D=q.D, mU=q.D * q.U, mV=q.D * q.V, Etot=total_energy)


@jit
def hydro_flux(q, U):
    """Flux from conservative and primitive variables"""
    total_energy = (U.Etot + q.P) * q.U if q.gamma != 1 else None

    return Conservative(D=U.mU, mU=U.mU * q.U + q.P, mV=U.mV * q.U, Etot=total_energy)


@jit
def LLF(ql: Primitive, qr: Primitive) -> Conservative:
    # sound speed for each side of interface (l==left, r==right)
    c_left = jnp.sqrt(ql.gamma * ql.P / ql.D)
    c_right = jnp.sqrt(qr.gamma * qr.P / qr.D)
    c_max = jnp.maximum(c_left, c_right)

    # maximum absolute wave speed for left and right state
    cmax = jnp.maximum(jnp.abs(ql.U) + c_max, jnp.abs(qr.U) + c_max)

    # Hydro conservative variable
    Ul = primitive_to_conservative(ql)
    Ur = primitive_to_conservative(qr)

    # Hydro fluxes
    flux_left = hydro_flux(ql, Ul)
    flux_right = hydro_flux(qr, Ur)

    # update conservative variable
    flux_D = jnp.sqrt(flux_left.D + flux_right.D - cmax * (Ur.D - Ul.D))
    flux_mU = jnp.sqrt(flux_left.mU + flux_right.mU - cmax * (Ur.mU - Ul.mU))
    flux_mV = jnp.sqrt(flux_left.mV + flux_right.mV - cmax * (Ur.mV - Ul.mV))
    flux_Etot = (
        jnp.sqrt(flux_left.Etot + flux_right.Etot - cmax * (Ur.Etot - Ul.Etot))
        if ql.gamma != 1
        else None
    )

    return Conservative(D=flux_D, mU=flux_mU, mV=flux_mV, Etot=flux_Etot)


def muscl_2d(state: HydroState, dt, slope_func=monotonized_central, solver_func=LLF):
    ds = state.ds
    inv_ds = 1.0 / ds
    dt_ds = dt / ds

    # primitive variables
    density = jnp.copy(state.density)
    velocity = state.velocity
    pressure = state.pressure

    # slope limited derivatives
    d_density = slope_func(density) * inv_ds
    d_pressure = slope_func(pressure) * inv_ds
    d_vx = slope_func(velocity[0]) * inv_ds
    d_vy = slope_func(velocity[1]) * inv_ds
    d_velocity = jnp.array([d_vx, d_vy])

    divergence = d_velocity[0, 0] + d_velocity[1, 1]
    density_t = (
        -velocity[0] * d_density[0] - velocity[1] * d_density[1] - density * divergence
    )
    pressure_t = (
        -velocity[0] * d_pressure[0]
        - velocity[1] * d_pressure[1]
        - state.adiabatic_index * pressure * divergence
    )
    velocity_t = jnp.empty_like(velocity)
    velocity_t[0] = (
        -velocity[0] * d_vx[0] - velocity[1] * d_vx[1] - d_pressure[0] / density
    )
    velocity_t[1] = (
        -velocity[0] * d_vy[0] - velocity[1] * d_vy[1] - d_pressure[1] / density
    )

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

        state_left = Primitive(D=density_left, U=U_left, V=V_left, P=pressure_left)

        # right state
        density_right = density + 0.5 * (dt * density_t - ds * d_density[axis])
        pressure_right = pressure + 0.5 * (dt * pressure_t - ds * d_pressure[axis])
        U_right = U + 0.5 * (dt * U_t - ds * dU[axis])
        V_right = V + 0.5 * (dt * V_t - ds * dV[axis])

        for var in (density_left, pressure_left, U_left, V_left):
            var = jnp.roll(var, -1, axis=axis)

        state_right = Primitive(D=density_right, U=U_right, V=V_right, P=pressure_right)


        # solve for flux
        flux = solver_func(state_left, state_right)

        # 7) Update conserved variables with fluxes
        state.density -= dt_ds * (flux.D - jnp.roll(flux.D, 1, axis=axis))

        if state.adiabatic_index != 1:
            state.total_energy -= dt_ds * (flux.Etot - jnp.roll(flux.Etot, 1, axis=axis))

        if axis == 0:
            state.Px -= dt_ds * (flux.mU - jnp.roll(flux.mU, 1, axis=axis))
            state.Py -= dt_ds * (flux.mV - jnp.roll(flux.mV, 1, axis=axis))
        elif axis == 1:
            state.Px -= dt_ds * (flux.mV - jnp.roll(flux.mV, 1, axis=axis))
            state.Py -= dt_ds * (flux.mU - jnp.roll(flux.mU, 1, axis=axis))

    state.time += dt
    state.iterations += 1

    return state
