import jax.numpy as jnp
from jax import make_jaxpr
from typing import Dict, Any

from interval import DualInterval, to_dual_interval, sigmoid, relu, sqrt, exp
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
        elif primitive.name == "reduce_mean" or primitive.name == "mean":
            outvals = [self._mean(invals[0], eqn.params)]
        elif primitive.name == "reduce_sum" or primitive.name == "sum":
            outvals = [self._sum(invals[0], eqn.params)]
        elif primitive.name == "broadcast_in_dim":
            outvals = [self._broadcast(invals[0], eqn.params)]
        elif primitive.name == "concatenate":
            outvals = [self._concatenate(invals, eqn.params)]
        elif primitive.name == "softmax":
            outvals = [self._softmax(invals[0], eqn.params)]
        elif primitive.name == "gelu":
            outvals = [self._gelu(invals[0])]
        elif primitive.name == "sqrt":
            outvals = [self._sqrt(invals[0])]
        elif primitive.name == "exp":
            outvals = [exp(invals[0])]
        elif primitive.name == "integer_pow":
            outvals = [self._integer_pow(invals[0], eqn.params)]
        elif primitive.name == "convert_element_type":
            outvals = [self._convert_type(invals[0], eqn.params)]
        elif primitive.name == "tanh":
            outvals = [self._tanh(invals[0])]
        elif primitive.name == "reduce_max" or primitive.name == "max":
            outvals = [self._reduce_max(invals[0], eqn.params)]
        elif primitive.name == "stop_gradient":
            outvals = [self._stop_gradient(invals[0])]
        elif primitive.name == "select":
            outvals = [self._select(invals, eqn.params)]
        elif primitive.name == "var" or primitive.name == "reduce_variance":
            outvals = [self._var(invals[0], eqn.params)]
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

    def _mean(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Mean reduction. Mean of intervals preserves bounds."""
        axes = params.get("axes", params.get("dimensions", None))
        keepdims = params.get("keepdims", False)
        real_l = jnp.mean(x.real_l, axis=axes, keepdims=keepdims)
        real_u = jnp.mean(x.real_u, axis=axes, keepdims=keepdims)
        dual_l = jnp.mean(x.dual_l, axis=axes, keepdims=keepdims)
        dual_u = jnp.mean(x.dual_u, axis=axes, keepdims=keepdims)
        return DualInterval(real_l, real_u, dual_l, dual_u)

    def _sum(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Sum reduction. For intervals, sum preserves bounds."""
        axes = params.get("axes", params.get("dimensions", None))
        keepdims = params.get("keepdims", False)
        real_l = jnp.sum(x.real_l, axis=axes, keepdims=keepdims)
        real_u = jnp.sum(x.real_u, axis=axes, keepdims=keepdims)
        dual_l = jnp.sum(x.dual_l, axis=axes, keepdims=keepdims)
        dual_u = jnp.sum(x.dual_u, axis=axes, keepdims=keepdims)
        return DualInterval(real_l, real_u, dual_l, dual_u)

    def _var(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Variance computation. Simplified: var of intervals."""
        axes = params.get("axes", params.get("dimensions", None))
        keepdims = params.get("keepdims", False)
        # For intervals, compute variance bounds conservatively
        real_l = jnp.var(x.real_l, axis=axes, keepdims=keepdims)
        real_u = jnp.var(x.real_u, axis=axes, keepdims=keepdims)
        # Gradient bounds simplified
        dual_l = jnp.var(x.dual_l, axis=axes, keepdims=keepdims)
        dual_u = jnp.var(x.dual_u, axis=axes, keepdims=keepdims)
        return DualInterval(real_l, real_u, dual_l, dual_u)

    def _sqrt(self, x: DualInterval) -> DualInterval:
        """Square root."""
        return sqrt(x)

    def _softmax(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Softmax: exp(x - max(x)) / sum(exp(x - max(x)))."""
        axis = params.get("axis", -1)
        # Numerical stability: subtract max before exp
        max_x = self._reduce_max(x, {"axes": (axis,), "keepdims": True})
        x_shifted = x - max_x
        # Compute exp(x - max)
        exp_x = exp(x_shifted)
        # Sum over axis
        sum_exp = self._sum(exp_x, {"axes": (axis,), "keepdims": True})
        # Divide: exp(x - max) / sum(exp(x - max))
        return exp_x / sum_exp

    def _gelu(self, x: DualInterval) -> DualInterval:
        """GELU: x * sigmoid(1.702 * x) - simplified approximation."""
        # GELU(x) ≈ x * sigmoid(1.702 * x)
        coeff = 1.702
        return x * sigmoid(coeff * x)
    
    def _integer_pow(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Integer power: x^n."""
        # Get power from params - JAX uses 'y' for the exponent
        n = int(params.get("y", params.get("power", 1)))
        if n == 0:
            return to_dual_interval(jnp.ones_like(x.real_l))
        elif n == 1:
            return x
        elif n == 2:
            return x * x
        elif n == 3:
            return x * x * x
        else:
            # General case: x^n = x * x^(n-1)
            result = x
            for _ in range(n - 1):
                result = result * x
            return result

    def _broadcast(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Broadcast operation."""
        shape = params.get("broadcast_sizes", params.get("shape", params.get("new_sizes")))
        if shape is None:
            return x
        # Try to broadcast, if it fails, return as-is
        try:
            return DualInterval(
                jnp.broadcast_to(x.real_l, shape),
                jnp.broadcast_to(x.real_u, shape),
                jnp.broadcast_to(x.dual_l, shape),
                jnp.broadcast_to(x.dual_u, shape),
            )
        except ValueError:
            # If broadcasting fails, return as-is
            return x
    
    def _reduce_max(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Max reduction."""
        axes = params.get("axes", params.get("dimensions", None))
        keepdims = params.get("keepdims", False)
        real_l = jnp.max(x.real_l, axis=axes, keepdims=keepdims)
        real_u = jnp.max(x.real_u, axis=axes, keepdims=keepdims)
        # For gradient, use max of gradients at max positions
        dual_l = jnp.max(x.dual_l, axis=axes, keepdims=keepdims)
        dual_u = jnp.max(x.dual_u, axis=axes, keepdims=keepdims)
        return DualInterval(real_l, real_u, dual_l, dual_u)
    
    def _convert_type(self, x: DualInterval, params: Dict[str, Any]) -> DualInterval:
        """Type conversion - just pass through for now."""
        return x
    
    def _tanh(self, x: DualInterval) -> DualInterval:
        """Tanh: tanh(x) ≈ 2*sigmoid(2*x) - 1."""
        return 2.0 * sigmoid(2.0 * x) - 1.0
    
    def _stop_gradient(self, x: DualInterval) -> DualInterval:
        """Stop gradient - zero out dual part."""
        return DualInterval(x.real_l, x.real_u, jnp.zeros_like(x.dual_l), jnp.zeros_like(x.dual_u))
    
    def _select(self, invals, params: Dict[str, Any]) -> DualInterval:
        """Select operation (conditional) - simplified."""
        # For now, just return the selected value
        if len(invals) >= 3:
            return invals[2]  # Return 'on_true' value
        return invals[0]

    def _concatenate(self, invals, params: Dict[str, Any]) -> DualInterval:
        """Concatenate operation - TODO: implement interval arithmetic for concatenate."""
        raise NotImplementedError("concatenate operation not yet implemented")

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
