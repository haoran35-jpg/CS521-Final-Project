import sys

sys.path.append("../src")

import jax.numpy as jnp
import jax.nn as nn
import numpy as np
from jax import value_and_grad

from interpreter import analyze_function

np.random.seed(42)


def simple_mlp(x, w1, b1, w2, b2):
    h = x @ w1 + b1  # TODO
    h = nn.sigmoid(h)
    y = h @ w2 + b2
    y = nn.sigmoid(y)
    return y


def create_mlp_params(input_dim, hidden_dim, output_dim):
    w1 = jnp.array(np.random.randn(input_dim, hidden_dim) * 0.1)
    b1 = jnp.array(np.random.randn(hidden_dim) * 0.1)
    w2 = jnp.array(np.random.randn(hidden_dim, output_dim) * 0.1)
    b2 = jnp.array(np.random.randn(output_dim) * 0.1)
    return w1, b1, w2, b2


def run_soundness_check():
    input_dim = 2
    hidden_dim = 3
    output_dim = 1

    w1, b1, w2, b2 = create_mlp_params(input_dim, hidden_dim, output_dim)

    x0 = jnp.array([0.5, -0.3])
    epsilon = 0.1

    def mlp_func(x):
        return simple_mlp(x, w1, b1, w2, b2)

    grad_seed = jnp.ones(input_dim)
    results = analyze_function(mlp_func, x0, epsilon=epsilon, gradient_seeds=[grad_seed])

    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    n_samples = 100

    samples = np.random.uniform(x0 - epsilon, x0 + epsilon, (n_samples, input_dim))

    def mlp_scalar(x):
        x_arr = jnp.array(x)
        return simple_mlp(x_arr, w1, b1, w2, b2)[0]

    value_and_grad_fn = value_and_grad(mlp_scalar)

    values = []
    gradients = []

    for sample in samples:
        val, grad_val = value_and_grad_fn(sample)
        values.append(val)
        gradients.append(grad_val)

    values = np.array(values)
    gradients = np.array(gradients)

    value_violations = 0
    grad_violations = 0

    for i, (val, grad_val) in enumerate(zip(values, gradients)):
        if not (val_l[0] <= val <= val_u[0]):
            value_violations += 1
            print(f"  Value violation at sample {i}: {val} not in [{val_l[0]}, {val_u[0]}]")

        for j in range(input_dim):
            if not (grad_l[j] <= grad_val[j] <= grad_u[j]):
                grad_violations += 1
                print(f"  Gradient violation at sample {i}, dim {j}: {grad_val[j]} not in [{grad_l[j]}, {grad_u[j]}]")

    if value_violations == 0 and grad_violations == 0:
        print("SUCCESS: All samples within computed bounds!")
    else:
        print("ERROR: Some violations detected!")


if __name__ == "__main__":
    run_soundness_check()
