import jax.numpy as jnp
import jax.nn as nn


class DualInterval:
    def __init__(
        self,
        real_l: jnp.ndarray,
        real_u: jnp.ndarray,
        dual_l=None,
        dual_u=None,
    ):
        self.real_l = real_l
        self.real_u = real_u

        if dual_l is None:
            dual_l = jnp.zeros_like(real_l)
        if dual_u is None:
            dual_u = jnp.zeros_like(real_l)

        self.dual_l = dual_l
        self.dual_u = dual_u

        assert jnp.all(real_l <= real_u), "Invalid real interval bounds"
        assert jnp.all(dual_l <= dual_u), "Invalid dual interval bounds"

    def __repr__(self):
        return f"Real: [{self.real_l}, {self.real_u}], Dual: [{self.dual_l}, {self.dual_u}]"

    def __neg__(self):
        return DualInterval(-self.real_u, -self.real_l, -self.dual_u, -self.dual_l)

    def __add__(self, other):
        if isinstance(other, DualInterval):
            return DualInterval(
                self.real_l + other.real_l,
                self.real_u + other.real_u,
                self.dual_l + other.dual_l,
                self.dual_u + other.dual_u,
            )
        else:  # Assume scalar
            return DualInterval(self.real_l + other, self.real_u + other, self.dual_l, self.dual_u)

    __radd__ = __add__

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, DualInterval):
            rl, ru = interval_mul(self.real_l, self.real_u, other.real_l, other.real_u)
            l1, u1 = interval_mul(self.real_l, self.real_u, other.dual_l, other.dual_u)
            l2, u2 = interval_mul(self.dual_l, self.dual_u, other.real_l, other.real_u)
            dual_l = l1 + l2
            dual_u = u1 + u2

            return DualInterval(rl, ru, dual_l, dual_u)
        else:  # Assume scalar
            if other >= 0:
                return DualInterval(self.real_l * other, self.real_u * other, self.dual_l * other, self.dual_u * other)
            else:
                return DualInterval(self.real_u * other, self.real_l * other, self.dual_u * other, self.dual_l * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DualInterval):
            one = jnp.ones_like(other.real_l)
            inv_real_l, inv_real_u = interval_div(one, one, other.real_l, other.real_u)

            squared_real_l, squared_real_u = interval_square(other.real_l, other.real_u)
            inv_dual_l, inv_dual_u = interval_div(other.dual_l, other.dual_u, squared_real_l, squared_real_u)
            inv_dual_l, inv_dual_u = -inv_dual_u, -inv_dual_l

            inv = DualInterval(inv_real_l, inv_real_u, inv_dual_l, inv_dual_u)
            return self * inv
        else:  # Assume scalar
            if other < 0:
                return DualInterval(self.real_u / other, self.real_l / other, self.dual_u / other, self.dual_l / other)
            else:
                return DualInterval(self.real_l / other, self.real_u / other, self.dual_l / other, self.dual_u / other)

    def get_bounds(self):
        return self.real_l, self.real_u

    def get_gradient_bounds(self):
        return self.dual_l, self.dual_u


def interval_mul(a_l, a_u, b_l, b_u):
    ac, ad, bc, bd = a_l * b_l, a_l * b_u, a_u * b_l, a_u * b_u
    real_l = jnp.minimum(jnp.minimum(ac, ad), jnp.minimum(bc, bd))
    real_u = jnp.maximum(jnp.maximum(ac, ad), jnp.maximum(bc, bd))
    return real_l, real_u


def interval_div(a_l, a_u, b_l, b_u):
    ac, ad, bc, bd = a_l / b_l, a_l / b_u, a_u / b_l, a_u / b_u
    real_l = jnp.minimum(jnp.minimum(ac, ad), jnp.minimum(bc, bd))
    real_u = jnp.maximum(jnp.maximum(ac, ad), jnp.maximum(bc, bd))
    return real_l, real_u


def interval_square(a_l, a_u):
    return interval_mul(a_l, a_u, a_l, a_u)


def relu(di: DualInterval) -> DualInterval:
    rl = jnp.maximum(0.0, di.real_l)
    ru = jnp.maximum(0.0, di.real_u)

    zero = jnp.zeros_like(di.dual_l)
    one = jnp.ones_like(di.dual_l)

    dual_l = jnp.where(di.real_u <= 0, zero, jnp.where(di.real_l >= 0, di.dual_l, jnp.minimum(zero, di.dual_l)))
    dual_u = jnp.where(
        di.real_u <= 0, zero, jnp.where(di.real_l >= 0, di.dual_u, jnp.maximum(one * di.dual_u, di.dual_u))
    )

    return DualInterval(rl, ru, dual_l, dual_u)


def sigmoid(di: DualInterval) -> DualInterval:
    rl = nn.sigmoid(di.real_l)
    ru = nn.sigmoid(di.real_u)

    deriv_l = rl * (1.0 - ru)
    deriv_u = ru * (1.0 - rl)

    dual_l, dual_u = interval_mul(deriv_l, deriv_u, di.dual_l, di.dual_u)

    return DualInterval(rl, ru, dual_l, dual_u)


def to_dual_interval(value, epsilon=0.0, gradient_seed=None):
    real_l = value - epsilon
    real_u = value + epsilon

    if gradient_seed is not None:
        dual_l = gradient_seed
        dual_u = gradient_seed
    else:
        dual_l = jnp.zeros_like(value)
        dual_u = jnp.zeros_like(value)

    return DualInterval(real_l, real_u, dual_l, dual_u)
