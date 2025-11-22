import jax.numpy as jnp
from jax import make_jaxpr
from typing import Dict, Any

from interval import DualInterval, to_dual_interval, sigmoid, relu
from operations import interval_matmul


def _is_literal(var):
    return hasattr(var, "val")


class JaxprDualIntervalInterpreter:
    # Default epsilon for debugging purposes
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
        self.env = {}

    def interpret(self, closed_jaxpr, *args, gradient_seeds=None):
        jaxpr = closed_jaxpr.jaxpr
        consts = closed_jaxpr.consts

        for const_var, const_val in zip(jaxpr.constvars, consts):
            self.env[const_var] = to_dual_interval(jnp.asarray(const_val))

        for i, (in_var, in_val) in enumerate(zip(jaxpr.invars, args)):
            grad_seed = gradient_seeds[i] if gradient_seeds else None
            self.env[in_var] = to_dual_interval(in_val, self.epsilon, grad_seed)

        for eqn in jaxpr.eqns:
            self._process_equation(eqn)

        return [self.env[outvar] for outvar in jaxpr.outvars]

    def _process_equation(self, eqn):
        primitive = eqn.primitive
        invals = [self._read(invar) for invar in eqn.invars]

        if primitive.name == "add":
            outvals = [invals[0] + invals[1]]
        elif primitive.name == "sub":
            outvals = [invals[0] - invals[1]]
        elif primitive.name == "mul":
            outvals = [invals[0] * invals[1]]
        elif primitive.name == "div":
            outvals = [invals[0] / invals[1]]
        elif primitive.name == "max":
            outvals = [self._max(invals[0], invals[1])]
        elif primitive.name == "relu":
            outvals = [relu(invals[0])]
        elif primitive.name == "logistic":
            outvals = [sigmoid(invals[0])]
        elif primitive.name == "custom_jvp_call":
            outvals = self._handle_custom_jvp(invals, eqn.params)
        elif primitive.name == "jit":
            outvals = self._handle_jit(invals, eqn.params)
        elif primitive.name == "reshape":
            outvals = [self._reshape(invals[0], eqn.params)]
        elif primitive.name == "transpose":
            outvals = [self._transpose(invals[0], eqn.params)]
        elif primitive.name == "dot_general" or primitive.name == "dot":
            outvals = [self._matmul(invals[0], invals[1], eqn.params)]
        else:
            raise NotImplementedError(f"Primitive {primitive.name} not supported")

        for outvar, outval in zip(eqn.outvars, outvals):
            self.env[outvar] = outval

    def _read(self, var):
        if _is_literal(var):
            return to_dual_interval(var.val, 0)
        elif var in self.env:
            return self.env[var]
        else:
            raise TypeError(f"Unknown variable type: {type(var)}")

    def _max(self, a: DualInterval, b: DualInterval) -> DualInterval:
        rl = jnp.maximum(a.real_l, b.real_l)
        ru = jnp.maximum(a.real_u, b.real_u)

        dual_l = jnp.minimum(a.dual_l, b.dual_l)
        dual_u = jnp.maximum(a.dual_u, b.dual_u)

        return DualInterval(rl, ru, dual_l, dual_u)

    def _reshape(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        new_shape = params.get("new_sizes", params.get("dimensions"))
        return DualInterval(
            x.real_l.reshape(new_shape),
            x.real_u.reshape(new_shape),
            x.dual_l.reshape(new_shape),
            x.dual_u.reshape(new_shape),
        )

    def _transpose(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        perm = params.get("permutation", (1, 0))
        return DualInterval(
            jnp.transpose(x.real_l, perm),
            jnp.transpose(x.real_u, perm),
            jnp.transpose(x.dual_l, perm),
            jnp.transpose(x.dual_u, perm),
        )

    def _matmul(self, a: DualInterval, b: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Matrix multiplication C = A @ B. Gradient: dC/dx = (dA/dx) @ B + A @ (dB/dx)."""
        real_l, real_u = interval_matmul(a.real_l, a.real_u, b.real_l, b.real_u)
        grad_a_b_l, grad_a_b_u = interval_matmul(a.dual_l, a.dual_u, b.real_l, b.real_u)
        a_grad_b_l, a_grad_b_u = interval_matmul(a.real_l, a.real_u, b.dual_l, b.dual_u)
        dual_l = grad_a_b_l + a_grad_b_l
        dual_u = grad_a_b_u + a_grad_b_u
        return DualInterval(real_l, real_u, dual_l, dual_u)

    def _handle_custom_jvp(self, invals, params):
        call_jaxpr = params["call_jaxpr"]
        return self._interpret_jaxpr(call_jaxpr, invals)

    def _handle_jit(self, invals, params):
        jaxpr = params["jaxpr"]
        return self._interpret_jaxpr(jaxpr, invals)

    def _interpret_jaxpr(self, jaxpr, invals):
        saved_env = self.env.copy()

        for invar, inval in zip(jaxpr.invars, invals):
            self.env[invar] = inval

        for eqn in jaxpr.eqns:
            self._process_equation(eqn)

        outvals = [self.env[outvar] for outvar in jaxpr.outvars]

        self.env = saved_env
        return outvals


def analyze_function(func, *args, epsilon=0.01, gradient_seeds=None):
    closed_jaxpr = make_jaxpr(func)(*args)

    print("=" * 60)
    print("JAXPR:")
    print("=" * 60)
    print(closed_jaxpr.jaxpr)
    print("=" * 60)

    interpreter = JaxprDualIntervalInterpreter(epsilon)
    return interpreter.interpret(closed_jaxpr, *args, gradient_seeds=gradient_seeds)
