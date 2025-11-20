import sys

sys.path.append("../src")

import jax.numpy as jnp
import jax.nn as nn
import numpy as np
from jax import value_and_grad

from interval import DualInterval, interval_mul, relu, sigmoid
from interpreter import analyze_function


def test_imul():
    print("\n--- Testing interval multiplication ---")
    l, u = interval_mul(jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0]), jnp.array([4.0]))
    assert np.isclose(l[0], 3.0) and np.isclose(u[0], 8.0)
    print("✓ [1, 2] * [3, 4] = [3, 8]")

    l, u = interval_mul(jnp.array([-2.0]), jnp.array([1.0]), jnp.array([3.0]), jnp.array([4.0]))
    assert np.isclose(l[0], -8.0) and np.isclose(u[0], 4.0)
    print("✓ [-2, 1] * [3, 4] = [-8, 4]")

    l, u = interval_mul(jnp.array([-2.0]), jnp.array([-1.0]), jnp.array([-4.0]), jnp.array([-3.0]))
    assert np.isclose(l[0], 3.0) and np.isclose(u[0], 8.0)
    print("✓ [-2, -1] * [-4, -3] = [3, 8]")


def test_dual_interval_operations():
    di1 = DualInterval(jnp.array([1.0]), jnp.array([2.0]), jnp.array([0.5]), jnp.array([1.0]))
    di2 = DualInterval(jnp.array([3.0]), jnp.array([4.0]), jnp.array([0.2]), jnp.array([0.3]))

    print("\n--- Testing dual-interval addition ---")
    result = di1 + di2
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 4.0) and np.isclose(val_u[0], 6.0)
    assert np.isclose(grad_l[0], 0.7) and np.isclose(grad_u[0], 1.3)
    print(f"✓ Addition: value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")

    print("\n--- Testing dual-interval multiplication ---")
    result = di1 * di2
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 3.0) and np.isclose(val_u[0], 8.0)
    print(f"✓ Value bounds: [{val_l[0]}, {val_u[0]}]")

    assert np.isclose(grad_l[0], 1.7) and np.isclose(grad_u[0], 4.6)
    print(f"✓ Gradient bounds: [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")


def test_subtraction_and_division():
    di1 = DualInterval(jnp.array([2.0]), jnp.array([4.0]), jnp.array([0.5]), jnp.array([1.0]))
    di2 = DualInterval(jnp.array([1.0]), jnp.array([2.0]), jnp.array([0.1]), jnp.array([0.2]))

    print("\n--- Testing dual-interval subtraction ---")
    result = di1 - di2
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 0.0) and np.isclose(val_u[0], 3.0)
    assert np.isclose(grad_l[0], 0.3) and np.isclose(grad_u[0], 0.9)
    print(f"✓ Subtraction: value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")

    print("\n--- Testing scalar subtraction ---")
    result = di1 - 1.0
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 1.0) and np.isclose(val_u[0], 3.0)
    assert np.isclose(grad_l[0], 0.5) and np.isclose(grad_u[0], 1.0)
    print(f"✓ Scalar subtraction: value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")

    print("\n--- Testing dual-interval division ---")
    di3 = DualInterval(jnp.array([4.0]), jnp.array([8.0]), jnp.array([0.4]), jnp.array([0.8]))
    di4 = DualInterval(jnp.array([2.0]), jnp.array([4.0]), jnp.array([0.1]), jnp.array([0.2]))
    result = di3 / di4
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 1.0) and np.isclose(val_u[0], 4.0)
    print(f"✓ Division value bounds: [{val_l[0]}, {val_u[0]}]")
    print(f"  Gradient bounds: [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- Testing scalar division ---")
    result = di3 / 2.0
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 2.0) and np.isclose(val_u[0], 4.0)
    assert np.isclose(grad_l[0], 0.2) and np.isclose(grad_u[0], 0.4)
    print(f"✓ Scalar division: value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")


def test_activation_functions():
    print("\n--- Testing ReLU activation ---")

    di_pos = DualInterval(jnp.array([1.0]), jnp.array([2.0]), jnp.array([0.5]), jnp.array([1.0]))
    result = relu(di_pos)
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 1.0) and np.isclose(val_u[0], 2.0)
    assert np.isclose(grad_l[0], 0.5) and np.isclose(grad_u[0], 1.0)
    print(f"✓ ReLU (positive): value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")

    di_neg = DualInterval(jnp.array([-2.0]), jnp.array([-1.0]), jnp.array([0.5]), jnp.array([1.0]))
    result = relu(di_neg)
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 0.0) and np.isclose(val_u[0], 0.0)
    assert np.isclose(grad_l[0], 0.0) and np.isclose(grad_u[0], 0.0)
    print(f"✓ ReLU (negative): value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]}, {grad_u[0]}]")

    di_cross = DualInterval(jnp.array([-1.0]), jnp.array([1.0]), jnp.array([0.5]), jnp.array([1.0]))
    result = relu(di_cross)
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    assert np.isclose(val_l[0], 0.0) and np.isclose(val_u[0], 1.0)
    print(f"✓ ReLU (crossing zero): value=[{val_l[0]}, {val_u[0]}], gradient=[{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- Testing Sigmoid activation ---")

    di = DualInterval(jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.5]), jnp.array([1.0]))
    result = sigmoid(di)
    val_l, val_u = result.get_bounds()
    grad_l, grad_u = result.get_gradient_bounds()

    expected_val_l = 1.0 / (1.0 + np.exp(0.0))
    expected_val_u = 1.0 / (1.0 + np.exp(-1.0))

    assert np.isclose(val_l[0], expected_val_l) and np.isclose(val_u[0], expected_val_u)
    print(f"✓ Sigmoid: value=[{val_l[0]:.3f}, {val_u[0]:.3f}], gradient=[{grad_l[0]:.3f}, {grad_u[0]:.3f}]")


def test_soundness_simple_functions():
    epsilon = 0.05

    print("\n--- f(x) = 2x + 3 ---")

    def linear(x):
        return 2 * x + 3

    x = jnp.array([1.0])
    grad_seed = jnp.array([1.0])
    results = analyze_function(linear, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def linear_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return (2 * x_arr + 3)[0]

    value_and_grad_f = value_and_grad(linear_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = x^2 ---")

    def quadratic(x):
        return x * x

    x = jnp.array([2.0])
    grad_seed = jnp.array([1.0])
    results = analyze_function(quadratic, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def quadratic_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return (x_arr * x_arr)[0]

    value_and_grad_f = value_and_grad(quadratic_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = max(0, x) ---")

    def relu(x):
        return jnp.maximum(0, x)

    x = jnp.array([0.0])
    grad_seed = jnp.array([1.0])
    results = analyze_function(relu, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def relu_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return jnp.maximum(0, x_arr)[0]

    value_and_grad_f = value_and_grad(relu_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = x^2 + 2x + 1 ---")

    def f(x):
        return x * x + 2 * x + 1

    x = jnp.array([1.0])
    epsilon = 0.1
    grad_seed = jnp.array([1.0])
    results = analyze_function(f, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def f_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return (x_arr * x_arr + 2 * x_arr + 1)[0]

    value_and_grad_f = value_and_grad(f_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 30)

    min_grad = float("inf")
    max_grad = float("-inf")

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        min_grad = min(min_grad, actual_grad)
        max_grad = max(max_grad, actual_grad)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert (
            grad_l <= actual_grad <= grad_u
        ), f"Gradient bound violated at x={test_x}: {actual_grad} not in [{grad_l[0]}, {grad_u[0]}]"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"Actual gradient range: [{min_grad:.3f}, {max_grad:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = x - 2 ---")

    def subtract(x):
        return x - 2

    x = jnp.array([3.0])
    epsilon = 0.05
    grad_seed = jnp.array([1.0])
    results = analyze_function(subtract, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def subtract_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return (x_arr - 2)[0]

    value_and_grad_f = value_and_grad(subtract_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = 10 / x ---")

    def divide(x):
        return 10.0 / x

    x = jnp.array([2.0])
    epsilon = 0.05
    grad_seed = jnp.array([1.0])
    results = analyze_function(divide, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def divide_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return (10.0 / x_arr)[0]

    value_and_grad_f = value_and_grad(divide_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = sigmoid(x) ---")

    def sigmoid_fn(x):
        return nn.sigmoid(x)

    x = jnp.array([0.0])
    epsilon = 0.1
    grad_seed = jnp.array([1.0])
    results = analyze_function(sigmoid_fn, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def sigmoid_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return nn.sigmoid(x_arr)[0]

    value_and_grad_f = value_and_grad(sigmoid_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")

    print("\n--- f(x) = relu(x) ---")

    def relu_fn(x):
        return nn.relu(x)

    x = jnp.array([0.0])
    epsilon = 0.1
    grad_seed = jnp.array([1.0])
    results = analyze_function(relu_fn, x, epsilon=epsilon, gradient_seeds=[grad_seed])
    val_l, val_u = results[0].get_bounds()
    grad_l, grad_u = results[0].get_gradient_bounds()

    def relu_scalar(x_scalar):
        x_arr = jnp.array([x_scalar])
        return nn.relu(x_arr)[0]

    value_and_grad_f = value_and_grad(relu_scalar)
    test_points = np.random.uniform(x[0] - epsilon, x[0] + epsilon, 20)

    for test_x in test_points:
        actual_val, actual_grad = value_and_grad_f(test_x)
        assert val_l <= actual_val <= val_u, f"Value bound violated at x={test_x}"
        assert grad_l <= actual_grad <= grad_u, f"Gradient bound violated at x={test_x}"

    print(f"✓ All test points within value bounds [{val_l[0]:.3f}, {val_u[0]:.3f}]")
    print(f"✓ All test points within gradient bounds [{grad_l[0]:.3f}, {grad_u[0]:.3f}]")


def run_all_tests():
    test_imul()
    test_dual_interval_operations()
    test_subtraction_and_division()
    test_activation_functions()
    test_soundness_simple_functions()


if __name__ == "__main__":
    run_all_tests()
