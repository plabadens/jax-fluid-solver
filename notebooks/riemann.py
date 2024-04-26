import chex
import jax.numpy as jnp
import jax.lax as lax
import numpy as onp

from jax import Array, jit, tree_util
from jax.typing import ArrayLike


def coordinates(n, Lbox=2.0 * jnp.pi) -> tuple[float, Array]:
    ds = Lbox / n
    # Two ways of accomplishing the same thing, either with arange or linspace
    # x = np.linspace(0., Lbox, num=n, endpoint=False)+0.5*ds
    x = ds * (jnp.arange(n) + 0.5)
    return ds, x


# Compute a "left slope". Returns the slope at the cell interface between x_i-1/2 and x_i+1/2
# at the index position i. Left slope is useful for computing slopes
@jit
def left_slope(f: ArrayLike, axis=0):
    return f - jnp.roll(f, 1, axis)


# Slopes are calculated at the grid center position
# Four slopes are given:
#   no_slope, Cen, MinMod, MonCen
#############################################################
# no_slope is zero everywhere producing the Upwind method
@jit
def no_slope(f):
    return jnp.zeros_like(f)


# Centered derivative; e.g. no slope limiter. This is unstable!
@jit
def Cen(f):
    ls = left_slope(f)  # left slope
    rs = jnp.roll(ls, -1)  # roll down once (giving the slope to the right)
    return 0.5 * (
        ls + rs
    )  # the average of the left and right slopes is simply the centered derivative (deriv(f))


# MinMod slope limiter
@jit
def MinMod(f):
    ls = left_slope(f)  # left slope
    rs = jnp.roll(ls, -1)  # right slope
    sign = jnp.ones(ls.shape)  # start with making sign equal +1
    sign[jnp.where(ls < 0.0)] -= 1.0  # where left slope is negative, make sign 0
    sign[
        jnp.where(rs < 0.0)
    ] -= 1.0  # if left *and* right slope are negative, sign is -1, if only right slope is negative, sign is 0
    return (
        jnp.minimum(jnp.abs(ls), jnp.abs(rs)) * sign
    )  # return the smallest size of the two slopes with correct sign, but zero if they disagree about sign


# MonCen slope limiter
@jit
def MonCen(f):
    size = jnp.shape(f)[0]

    ls = left_slope(f)  # left slope
    rs = jnp.roll(ls, -1)  # right slope
    cs = jnp.zeros(size)  # MonCen starts out as zero slope
    w = jnp.where(
        ls * rs > 0.0, size=size
    )  # Where both slopes agree in sign, we compute it
    cs = cs.at[w].set(
        2.0 * ls[w] * rs[w] / (ls[w] + rs[w])
    )  # MonCen slope is the harmonic average of the left and right state
    return cs


# Define an hydro state including all auxiliary scalars and coordinates in a class
class HydroState(object):
    def __init__(self, n, gamma=1.0, cs=1.0, Lbox=2.0 * jnp.pi):
        dx, x = coordinates(n, Lbox=Lbox)
        self.n = n  # number of points
        self.dx = dx  # cell size
        self.Lbox = Lbox  # Box size
        self.x = x  # coordinate axis
        self.gamma = gamma  # adiabatic index
        self.cs = cs  # initial sound speed, if isothermal it is fixed
        self.t = 0.0  # time
        self.rho = jnp.ones(n)  # density
        self.Px = jnp.zeros(n)  # momentum density = rho*velocity_x
        if gamma != 1.0:  # non-isothermal
            Pressure = cs**2 * self.rho / gamma
            Eint = Pressure / (self.gamma - 1.0)
            self.Etot = Eint  # total energy

    # Compute velocity from state vector
    def velocity(self):
        """Compute velocity from conservative variables"""
        return self.Px / self.rho

    # Compute pressure from state vector
    def pressure(self):
        """Compute pressure from conservative variables"""
        if self.gamma == 1.0:
            P = self.cs**2 * self.rho
        else:
            Eint = self.Etot - 0.5 * self.Px**2 / self.rho  # Internal energy
            P = (self.gamma - 1.0) * Eint
        return P

    def sound_speed(self):
        """Sound speed for HD"""
        # if gamma=1 gas is isothermal, and sound speed is a property of the equation of state
        if self.gamma == 1.0:
            return self.cs
        else:
            P = self.pressure()
            cs = jnp.sqrt(self.gamma * P / self.rho)
            return cs

    # Courant condition with default Courant number=0.2 for a fluid
    # maximum propagation velocity is max(|v| + sound speed), where max is taken over all cells
    def Courant(self, Cdt=0.2):
        """Courant condition for HD"""
        speed = abs(self.velocity())
        dt = Cdt * self.dx / jnp.max(speed + self.sound_speed())
        return dt

@chex.dataclass
class Primitive:
    D: Array
    U: Array
    P: Array
    gamma: float = 1.0


@chex.dataclass
class Conservative:
    D: Array
    mU: Array


# Conservative variables computed from primitve variable
@jit
def primitive_to_conservative(q: Primitive) -> Conservative:
    return Conservative(D=q.D, mU=q.D * q.U)


# Hydro flux from conservative and primitive variables
@jit
def hydro_flux(q, U):
    return Conservative(D=U.mU, mU=U.mU * q.U + q.P)


# LLF is the most diffuse Riemann solver. But also the most stable.
# ql = (density, velocity, pressure) = (D, U, P), qr are state vectors for the _primitive_ variables
@jit
def LLF(ql, qr):
    # sound speed for each side of interface (l==left, r==right)
    c_left = (ql.gamma * ql.P / ql.D) ** 0.5
    c_right = (qr.gamma * qr.P / qr.D) ** 0.5
    c_max = jnp.maximum(c_left, c_right)

    # maximum absolute wave speed for left and right state
    cmax = jnp.maximum(jnp.abs(ql.U) + c_max, jnp.abs(qr.U) + c_max)

    # Hydro conservative variable
    Ul = primitive_to_conservative(ql)
    Ur = primitive_to_conservative(qr)

    # Hydro fluxes
    Fl = hydro_flux(ql, Ul)
    Fr = hydro_flux(qr, Ur)

    # LLF flux based on maximum wavespeed.
    # The general form is "(F_left + F_right - cmax*(U_right - U_left)) / 2"
    # where U is the state vector of the conserved variables
    flux_D = 0.5 * (Fl.D + Fr.D - cmax * (Ur.D - Ul.D))
    flux_mU = 0.5 * (Fl.mU + Fr.mU - cmax * (Ur.mU - Ul.mU))
    # Flux.Etot = ...

    return Conservative(D=flux_D, mU=flux_mU)


# HLL (Harten, Lax, van Leer) is a bit less diffuse. We compute individual wave speeds for each state
# ql, qr are state vectors for the _primitive_ variables
@jit
def HLL(ql, qr):
    # sound speed for each side of interface (l==left, r==right)
    c_left = (ql.gamma * ql.P / ql.D) ** 0.5
    c_right = (qr.gamma * qr.P / qr.D) ** 0.5
    c_max = jnp.maximum(c_left, c_right)

    # maximum wave speeds to the left and right (guaranteed to have right sign)
    SL = jnp.minimum(jnp.minimum(ql.U, qr.U) - c_max, 0)  # <= 0.
    SR = jnp.maximum(jnp.maximum(ql.U, qr.U) + c_max, 0)  # >= 0.

    # Hydro conservative variable
    Ul = primitive_to_conservative(ql)
    Ur = primitive_to_conservative(qr)

    # Hydro fluxes
    Fl = hydro_flux(ql, Ul)
    Fr = hydro_flux(qr, Ur)

    # HLL flux based on wavespeeds. If SL < 0 and SR > 0 then mix state appropriately
    # The general form is
    #    (SR * F_left - SL * F_right + SL * SR *(U_right - U_left)) / (SR - SL)
    # where U is the state vector of the conserved variables
    flux_D = (SR * Fl.D - SL * Fr.D + SL * SR * (Ur.D - Ul.D)) / (SR - SL)
    flux_mU = (SR * Fl.mU - SL * Fr.mU + SL * SR * (Ur.mU - Ul.mU)) / (SR - SL)
    # Flux.Etot = ...

    return Conservative(D=flux_D, mU=flux_mU)


# Hydrodynamics solver based on MUSCL scheme
def muscl(u: HydroState, dt, Slope=MinMod, Riemann_Solver=HLL):
    dx = u.dx
    idx = 1.0 / u.dx
    dtdx = dt / u.dx

    # 1) Compute primitive variables at cell center (rho, v, P)
    rho = u.rho
    v = u.velocity()
    P = u.pressure()

    # 2) Compute slope limited derivatives based on centered points
    drhodx = Slope(rho) * idx
    dvdx = Slope(v) * idx
    dPdx = Slope(P) * idx

    # 3) Trace forward to find solution at [t+dt/2, x +- dx/2]
    # Time evolution for source terms
    rho_t = -v * drhodx - dvdx * rho
    v_t = -v * dvdx - dPdx / rho
    # Here we need to compute time evolution for P

    # Spatial interpolation + time terms
    # left state at t + dt/2 -- AS SEEN FROM THE INTERFACE
    ql_D = rho + 0.5 * (dt * rho_t + dx * drhodx)
    ql_U = v + 0.5 * (dt * v_t + dx * dvdx)
    ql_P = u.cs**2 * ql_D
    ql = Primitive(D=ql_D, U=ql_U, P=ql_P, gamma=u.gamma)

    # Spatial interpolation + time terms
    # right state at t + dt/2 -- AS SEEN FROM THE INTERFACE
    qr_D = rho + 0.5 * (dt * rho_t - dx * drhodx)
    qr_U = v + 0.5 * (dt * v_t - dx * dvdx)
    qr_P = u.cs**2 * qr_D

    qr = Primitive(D=qr_D, U=qr_U, P=qr_P, gamma=u.gamma)

    # make sure that right state is centered correctly.
    # Numerical index follow upstaggered interfaces,
    qr.D = jnp.roll(qr.D, -1)
    qr.U = jnp.roll(qr.U, -1)
    qr.P = jnp.roll(qr.P, -1)

    # 4) Solve for flux based on interface values
    Flux = Riemann_Solver(ql, qr)

    # 5) Update conserved variables.
    #    From the cell center (at x_i) point of view:
    #       * 1st term is the upstaggered value at the interface position x_i + dx/2
    #       * 2nd term is the downstaggered value at the interface position x_i - dx/2. Therefore we roll.
    u.rho += -dtdx * (Flux.D - jnp.roll(Flux.D, 1))
    u.Px += -dtdx * (Flux.mU - jnp.roll(Flux.mU, 1))
    # Update energy flux ...

    return u
