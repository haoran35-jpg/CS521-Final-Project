"""
Visualize value bounds and dual bounds during GPT2 model execution.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax.numpy as jnp
from jax import make_jaxpr
from interpreter import JaxprDualIntervalInterpreter


class BoundsTracker:
    """Track bounds during interpretation."""
    def __init__(self):
        self.steps = []
        self.value_bounds = []  # (min, max) for each step
        self.dual_bounds = []   # (min, max) for each step
        self.operation_names = []
        # Special tracking for matmul
        self.matmul_steps = []
        self.matmul_value_bounds = []
        self.matmul_dual_bounds = []
        self.matmul_operation_names = []
        
    def record(self, name, result):
        """Record bounds after an operation."""
        if isinstance(result, tuple):
            result = result[0]
        
        val_l, val_u = result.get_bounds()
        grad_l, grad_u = result.get_gradient_bounds()
        
        # Compute statistics
        val_min = float(jnp.min(val_l))
        val_max = float(jnp.max(val_u))
        grad_min = float(jnp.min(grad_l))
        grad_max = float(jnp.max(grad_u))
        
        step = len(self.steps)
        self.steps.append(step)
        self.value_bounds.append((val_min, val_max))
        self.dual_bounds.append((grad_min, grad_max))
        self.operation_names.append(name)
        
        # Special tracking for matmul
        if name in ["dot_general", "dot", "matmul"]:
            self.matmul_steps.append(len(self.matmul_steps))
            self.matmul_value_bounds.append((val_min, val_max))
            self.matmul_dual_bounds.append((grad_min, grad_max))
            self.matmul_operation_names.append(f"{name}_{len(self.matmul_steps)}")


class TrackingInterpreter(JaxprDualIntervalInterpreter):
    """Interpreter that tracks bounds."""
    def __init__(self, epsilon=0.01, tracker=None):
        super().__init__(epsilon)
        self.tracker = tracker
        
    def _process_equation(self, eqn):
        primitive = eqn.primitive
        invals = [self._read(invar) for invar in eqn.invars]
        
        # Process as normal
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
            from interval import relu
            outvals = [relu(invals[0])]
        elif primitive.name == "logistic":
            from interval import sigmoid
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
        elif primitive.name == "softmax":
            outvals = [self._softmax(invals[0], eqn.params)]
        elif primitive.name == "gelu":
            outvals = [self._gelu(invals[0])]
        elif primitive.name == "sqrt":
            outvals = [self._sqrt(invals[0])]
        elif primitive.name == "exp":
            from interval import exp
            outvals = [exp(invals[0])]
        else:
            # Skip unsupported operations for visualization
            outvals = invals[:len(eqn.outvars)]
        
        # Track bounds for each output
        if self.tracker:
            for i, outval in enumerate(outvals):
                if hasattr(outval, 'get_bounds'):
                    op_name = primitive.name
                    if len(outvals) > 1:
                        op_name = f"{primitive.name}_{i}"
                    self.tracker.record(op_name, outval)
        
        for outvar, outval in zip(eqn.outvars, outvals):
            self.env[outvar] = outval


def visualize_simple_gpt2():
    """Visualize bounds for a simple GPT2 forward pass."""
    print("Running simple GPT2 model and tracking bounds...")
    
    def simple_gpt2(x, w_embed, w_pos, w_out):
        # Token embedding
        x_embed = jnp.matmul(x, w_embed)
        # Position embedding (simplified)
        x_pos = jnp.matmul(x, w_pos)
        # Combine
        x = x_embed + x_pos
        # Output projection
        return jnp.matmul(x, w_out)
    
    batch_size = 2
    seq_len = 4
    vocab_size = 100
    d_model = 8
    
    x = jnp.array(np.random.randn(batch_size, seq_len, vocab_size))
    w_embed = jnp.array(np.random.randn(vocab_size, d_model))
    w_pos = jnp.array(np.random.randn(vocab_size, d_model))
    w_out = jnp.array(np.random.randn(d_model, vocab_size))
    
    # Create tracker and interpreter
    tracker = BoundsTracker()
    epsilon = 0.01
    
    # Create gradient seeds for tracking gradients
    grad_seed_x = jnp.ones_like(x) * 0.1
    grad_seed_w_embed = jnp.ones_like(w_embed) * 0.05
    grad_seed_w_pos = jnp.ones_like(w_pos) * 0.05
    grad_seed_w_out = jnp.ones_like(w_out) * 0.05
    
    interpreter = TrackingInterpreter(epsilon, tracker)
    closed_jaxpr = make_jaxpr(simple_gpt2)(x, w_embed, w_pos, w_out)
    
    # Run interpretation with gradient seeds
    try:
        results = interpreter.interpret(
            closed_jaxpr, x, w_embed, w_pos, w_out,
            gradient_seeds=[grad_seed_x, grad_seed_w_embed, grad_seed_w_pos, grad_seed_w_out]
        )
        print(f"✓ Model executed successfully")
        print(f"  Total operations tracked: {len(tracker.steps)}")
    except Exception as e:
        print(f"⚠ Error during execution: {e}")
        return
    
    # Plot results
    plot_bounds(tracker, "Simple GPT2 Forward Pass")


def visualize_feed_forward():
    """Visualize bounds for feed-forward network."""
    print("\nRunning feed-forward network and tracking bounds...")
    
    def feed_forward(x, w1, w2):
        h = jnp.matmul(x, w1)
        h = jnp.maximum(0, h)  # ReLU
        return jnp.matmul(h, w2)
    
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 32
    
    x = jnp.array(np.random.randn(batch_size, seq_len, d_model))
    w1 = jnp.array(np.random.randn(d_model, d_ff))
    w2 = jnp.array(np.random.randn(d_ff, d_model))
    
    tracker = BoundsTracker()
    epsilon = 0.01
    
    # Create gradient seeds
    grad_seed_x = jnp.ones_like(x) * 0.1
    grad_seed_w1 = jnp.ones_like(w1) * 0.05
    grad_seed_w2 = jnp.ones_like(w2) * 0.05
    
    interpreter = TrackingInterpreter(epsilon, tracker)
    closed_jaxpr = make_jaxpr(feed_forward)(x, w1, w2)
    
    try:
        results = interpreter.interpret(
            closed_jaxpr, x, w1, w2,
            gradient_seeds=[grad_seed_x, grad_seed_w1, grad_seed_w2]
        )
        print(f"✓ Feed-forward executed successfully")
        print(f"  Total operations tracked: {len(tracker.steps)}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        return
    
    plot_bounds(tracker, "Feed-Forward Network")


def visualize_multi_layer_mlp():
    """Visualize bounds for multi-layer MLP."""
    print("\nRunning multi-layer MLP and tracking bounds...")
    
    def multi_layer_mlp(x, w1, w2, w3, w4):
        # Layer 1
        h1 = jnp.matmul(x, w1)
        h1 = jnp.maximum(0, h1)  # ReLU
        # Layer 2
        h2 = jnp.matmul(h1, w2)
        h2 = jnp.maximum(0, h2)  # ReLU
        # Layer 3
        h3 = jnp.matmul(h2, w3)
        h3 = jnp.maximum(0, h3)  # ReLU
        # Layer 4
        return jnp.matmul(h3, w4)
    
    batch_size = 4
    seq_len = 8
    d_model = 16
    d_hidden1 = 64
    d_hidden2 = 32
    d_hidden3 = 16
    
    x = jnp.array(np.random.randn(batch_size, seq_len, d_model))
    w1 = jnp.array(np.random.randn(d_model, d_hidden1))
    w2 = jnp.array(np.random.randn(d_hidden1, d_hidden2))
    w3 = jnp.array(np.random.randn(d_hidden2, d_hidden3))
    w4 = jnp.array(np.random.randn(d_hidden3, d_model))
    
    tracker = BoundsTracker()
    epsilon = 0.01
    
    grad_seed_x = jnp.ones_like(x) * 0.1
    grad_seed_w1 = jnp.ones_like(w1) * 0.05
    grad_seed_w2 = jnp.ones_like(w2) * 0.05
    grad_seed_w3 = jnp.ones_like(w3) * 0.05
    grad_seed_w4 = jnp.ones_like(w4) * 0.05
    
    interpreter = TrackingInterpreter(epsilon, tracker)
    closed_jaxpr = make_jaxpr(multi_layer_mlp)(x, w1, w2, w3, w4)
    
    try:
        results = interpreter.interpret(
            closed_jaxpr, x, w1, w2, w3, w4,
            gradient_seeds=[grad_seed_x, grad_seed_w1, grad_seed_w2, grad_seed_w3, grad_seed_w4]
        )
        print(f"✓ Multi-layer MLP executed successfully")
        print(f"  Total operations tracked: {len(tracker.steps)}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        return
    
    plot_bounds(tracker, "Multi-Layer MLP")


def visualize_complex_gpt2():
    """Visualize bounds for a more complex GPT2-like model."""
    print("\nRunning complex GPT2 model and tracking bounds...")
    
    def complex_gpt2(x, w_embed, w_pos, w1, w2, w3, w_out):
        # Embedding
        x_embed = jnp.matmul(x, w_embed)
        x_pos = jnp.matmul(x, w_pos)
        x = x_embed + x_pos
        
        # Multiple transformations
        h1 = jnp.matmul(x, w1)
        h1 = jnp.maximum(0, h1)  # ReLU
        h2 = jnp.matmul(h1, w2)
        h2 = jnp.maximum(0, h2)  # ReLU
        h3 = jnp.matmul(h2, w3)
        h3 = jnp.maximum(0, h3)  # ReLU
        
        # Output
        return jnp.matmul(h3, w_out)
    
    batch_size = 4
    seq_len = 8
    vocab_size = 200
    d_model = 32
    d_hidden = 128
    
    x = jnp.array(np.random.randn(batch_size, seq_len, vocab_size))
    w_embed = jnp.array(np.random.randn(vocab_size, d_model))
    w_pos = jnp.array(np.random.randn(vocab_size, d_model))
    w1 = jnp.array(np.random.randn(d_model, d_hidden))
    w2 = jnp.array(np.random.randn(d_hidden, d_hidden))
    w3 = jnp.array(np.random.randn(d_hidden, d_model))
    w_out = jnp.array(np.random.randn(d_model, vocab_size))
    
    tracker = BoundsTracker()
    epsilon = 0.01
    
    grad_seeds = [
        jnp.ones_like(x) * 0.1,
        jnp.ones_like(w_embed) * 0.05,
        jnp.ones_like(w_pos) * 0.05,
        jnp.ones_like(w1) * 0.05,
        jnp.ones_like(w2) * 0.05,
        jnp.ones_like(w3) * 0.05,
        jnp.ones_like(w_out) * 0.05,
    ]
    
    interpreter = TrackingInterpreter(epsilon, tracker)
    closed_jaxpr = make_jaxpr(complex_gpt2)(x, w_embed, w_pos, w1, w2, w3, w_out)
    
    try:
        results = interpreter.interpret(
            closed_jaxpr, x, w_embed, w_pos, w1, w2, w3, w_out,
            gradient_seeds=grad_seeds
        )
        print(f"✓ Complex GPT2 executed successfully")
        print(f"  Total operations tracked: {len(tracker.steps)}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        return
    
    plot_bounds(tracker, "Complex GPT2 Model")


def plot_matmul_bounds(tracker, title):
    """Plot matmul-specific bounds visualization."""
    if len(tracker.matmul_steps) == 0:
        return
    
    steps = tracker.matmul_steps
    val_mins = [b[0] for b in tracker.matmul_value_bounds]
    val_maxs = [b[1] for b in tracker.matmul_value_bounds]
    grad_mins = [b[0] for b in tracker.matmul_dual_bounds]
    grad_maxs = [b[1] for b in tracker.matmul_dual_bounds]
    
    # Calculate bound widths
    val_widths = [max_val - min_val for min_val, max_val in zip(val_mins, val_maxs)]
    grad_widths = [max_val - min_val for min_val, max_val in zip(grad_mins, grad_maxs)]
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])
    
    # Plot value bounds for matmul
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(steps, val_mins, val_maxs, alpha=0.3, color='green', label='MatMul Value Bounds')
    ax1.plot(steps, val_mins, 'g-', linewidth=3, label='Value Min', marker='o', markersize=6)
    ax1.plot(steps, val_maxs, 'g--', linewidth=3, label='Value Max', marker='s', markersize=6)
    ax1.set_ylabel('Value Bound', fontsize=12)
    ax1.set_title(f'{title} - MatMul Value Bounds (Important Operation)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add matmul labels
    for i, (step, op_name) in enumerate(zip(steps, tracker.matmul_operation_names)):
        ax1.text(step, val_maxs[i] + (val_maxs[i] - val_mins[i]) * 0.15, 
                op_name, rotation=45, ha='left', fontsize=9, fontweight='bold', alpha=0.8)
    
    # Plot dual bounds for matmul
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(steps, grad_mins, grad_maxs, alpha=0.3, color='orange', label='MatMul Dual Bounds')
    ax2.plot(steps, grad_mins, 'orange', linewidth=3, label='Dual Min', marker='o', markersize=6)
    ax2.plot(steps, grad_maxs, 'orange', linestyle='--', linewidth=3, label='Dual Max', marker='s', markersize=6)
    ax2.set_ylabel('Dual Bound (Gradient)', fontsize=12)
    ax2.set_title(f'{title} - MatMul Dual Bounds (Gradients)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot bound widths for matmul
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, val_widths, 'g-', linewidth=3, label='MatMul Value Width', marker='o', markersize=6)
    ax3.plot(steps, grad_widths, 'orange', linewidth=3, label='MatMul Dual Width', marker='s', markersize=6)
    ax3.set_xlabel('MatMul Operation Index', fontsize=12)
    ax3.set_ylabel('Bound Width', fontsize=12)
    ax3.set_title('MatMul Bound Widths (Uncertainty)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = title.lower().replace(' ', '_').replace('-', '_') + '_matmul_bounds.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  MatMul-specific plot saved to: {filename}")
    
    # Print matmul statistics
    print(f"  MatMul operations: {len(steps)}")
    print(f"  MatMul value bounds: [{min(val_mins):.3f}, {max(val_maxs):.3f}]")
    print(f"  MatMul dual bounds: [{min(grad_mins):.3f}, {max(grad_maxs):.3f}]")
    print(f"  Max MatMul value width: {max(val_widths):.3f}")
    print(f"  Max MatMul dual width: {max(grad_widths):.3f}")
    
    plt.close()


def visualize_attention_like():
    """Visualize bounds for attention-like operations."""
    print("\nRunning attention-like model and tracking bounds...")
    
    def attention_like(x, wq, wk, wv, wo):
        # Query, Key, Value
        q = jnp.matmul(x, wq)
        k = jnp.matmul(x, wk)
        v = jnp.matmul(x, wv)
        
        # Attention scores (simplified)
        scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)))
        scores = scores / jnp.sqrt(float(q.shape[-1]))
        
        # Apply ReLU as attention weights (simplified)
        attn_weights = jnp.maximum(0, scores)
        
        # Apply attention
        attn_out = jnp.matmul(attn_weights, v)
        
        # Output projection
        return jnp.matmul(attn_out, wo)
    
    batch_size = 2
    seq_len = 6
    d_model = 16
    d_head = 16
    
    x = jnp.array(np.random.randn(batch_size, seq_len, d_model))
    wq = jnp.array(np.random.randn(d_model, d_head))
    wk = jnp.array(np.random.randn(d_model, d_head))
    wv = jnp.array(np.random.randn(d_model, d_head))
    wo = jnp.array(np.random.randn(d_head, d_model))
    
    tracker = BoundsTracker()
    epsilon = 0.01
    
    grad_seeds = [
        jnp.ones_like(x) * 0.1,
        jnp.ones_like(wq) * 0.05,
        jnp.ones_like(wk) * 0.05,
        jnp.ones_like(wv) * 0.05,
        jnp.ones_like(wo) * 0.05,
    ]
    
    interpreter = TrackingInterpreter(epsilon, tracker)
    closed_jaxpr = make_jaxpr(attention_like)(x, wq, wk, wv, wo)
    
    try:
        results = interpreter.interpret(
            closed_jaxpr, x, wq, wk, wv, wo,
            gradient_seeds=grad_seeds
        )
        print(f"✓ Attention-like model executed successfully")
        print(f"  Total operations tracked: {len(tracker.steps)}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        return
    
    plot_bounds(tracker, "Attention-Like Model")


def plot_bounds(tracker, title):
    """Plot value bounds and dual bounds over operations."""
    if len(tracker.steps) == 0:
        print("No data to plot")
        return
    
    steps = tracker.steps
    val_mins = [b[0] for b in tracker.value_bounds]
    val_maxs = [b[1] for b in tracker.value_bounds]
    grad_mins = [b[0] for b in tracker.dual_bounds]
    grad_maxs = [b[1] for b in tracker.dual_bounds]
    
    # Calculate bound widths
    val_widths = [max_val - min_val for min_val, max_val in zip(val_mins, val_maxs)]
    grad_widths = [max_val - min_val for min_val, max_val in zip(grad_mins, grad_maxs)]
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])
    
    # Plot value bounds
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(steps, val_mins, val_maxs, alpha=0.3, color='blue', label='Value Bounds')
    ax1.plot(steps, val_mins, 'b-', linewidth=2, label='Value Min', marker='o', markersize=4)
    ax1.plot(steps, val_maxs, 'b--', linewidth=2, label='Value Max', marker='s', markersize=4)
    ax1.set_ylabel('Value Bound', fontsize=12)
    ax1.set_title(f'{title} - Value Bounds', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add operation labels
    for i, (step, op_name) in enumerate(zip(steps, tracker.operation_names)):
        if i % max(1, len(steps) // 10) == 0 or i == len(steps) - 1:
            ax1.text(step, val_maxs[i] + (val_maxs[i] - val_mins[i]) * 0.1, 
                    op_name, rotation=45, ha='left', fontsize=8, alpha=0.7)
    
    # Plot dual bounds
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(steps, grad_mins, grad_maxs, alpha=0.3, color='red', label='Dual Bounds')
    ax2.plot(steps, grad_mins, 'r-', linewidth=2, label='Dual Min', marker='o', markersize=4)
    ax2.plot(steps, grad_maxs, 'r--', linewidth=2, label='Dual Max', marker='s', markersize=4)
    ax2.set_ylabel('Dual Bound (Gradient)', fontsize=12)
    ax2.set_title(f'{title} - Dual Bounds (Gradients)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot bound widths
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, val_widths, 'b-', linewidth=2, label='Value Width', marker='o', markersize=4)
    ax3.plot(steps, grad_widths, 'r-', linewidth=2, label='Dual Width', marker='s', markersize=4)
    ax3.set_xlabel('Operation Step', fontsize=12)
    ax3.set_ylabel('Bound Width', fontsize=12)
    ax3.set_title('Bound Widths (Uncertainty)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = title.lower().replace(' ', '_').replace('-', '_') + '_bounds.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {filename}")
    
    # Print summary statistics
    print(f"  Value bounds: [{min(val_mins):.3f}, {max(val_maxs):.3f}]")
    print(f"  Dual bounds: [{min(grad_mins):.3f}, {max(grad_maxs):.3f}]")
    print(f"  Max value width: {max(val_widths):.3f}")
    print(f"  Max dual width: {max(grad_widths):.3f}")
    
    plt.close()
    
    # Plot matmul-specific visualization if available
    if len(tracker.matmul_steps) > 0:
        plot_matmul_bounds(tracker, title)


if __name__ == "__main__":
    print("=" * 60)
    print("GPT2 Bounds Visualization")
    print("=" * 60)
    
    visualize_simple_gpt2()
    visualize_feed_forward()
    visualize_multi_layer_mlp()
    visualize_complex_gpt2()
    visualize_attention_like()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

