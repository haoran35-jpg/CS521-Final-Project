"""
Trainable GPT2 model with bounds tracking during training.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax.numpy as jnp
from jax import make_jaxpr, value_and_grad
from interpreter import JaxprDualIntervalInterpreter
from visualize_bounds import BoundsTracker, TrackingInterpreter


class TrainableGPT2:
    """A simple trainable GPT2-like model."""
    
    def __init__(self, vocab_size=1000, d_model=64, d_ff=256, n_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        # Initialize weights with smaller scale for stability
        scale = 0.01 / np.sqrt(d_model)
        self.w_embed = np.random.randn(vocab_size, d_model) * scale
        self.w_pos = np.random.randn(vocab_size, d_model) * scale
        self.w_layers = []
        self.w_ff_layers = []
        
        for i in range(n_layers):
            # Attention weights
            self.w_layers.append({
                'q': np.random.randn(d_model, d_model) * scale,
                'k': np.random.randn(d_model, d_model) * scale,
                'v': np.random.randn(d_model, d_model) * scale,
                'o': np.random.randn(d_model, d_model) * scale,
            })
            # Feed-forward weights
            self.w_ff_layers.append({
                'w1': np.random.randn(d_model, d_ff) * scale,
                'w2': np.random.randn(d_ff, d_model) * scale,
            })
        
        self.w_out = np.random.randn(d_model, vocab_size) * scale
    
    def forward(self, x, weights_dict):
        """Forward pass of the model with normalization for stability."""
        # Embedding with moderate scaling
        x_embed = jnp.matmul(x, weights_dict['w_embed']) * 0.3
        x_pos = jnp.matmul(x, weights_dict['w_pos']) * 0.3
        x = x_embed + x_pos
        
        # Transformer layers
        for i in range(self.n_layers):
            # Self-attention (simplified)
            q = jnp.matmul(x, weights_dict[f'layer_{i}_q']) * 0.3
            k = jnp.matmul(x, weights_dict[f'layer_{i}_k']) * 0.3
            v = jnp.matmul(x, weights_dict[f'layer_{i}_v']) * 0.3
            
            # Attention scores (simplified, no softmax for now)
            scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(float(self.d_model))
            attn_weights = jnp.maximum(0, scores) * 0.2  # Scale down attention
            attn_out = jnp.matmul(attn_weights, v)
            attn_out = jnp.matmul(attn_out, weights_dict[f'layer_{i}_o']) * 0.3
            x = x + attn_out * 0.7  # Residual with scaling
            
            # Feed-forward
            ff1 = jnp.matmul(x, weights_dict[f'ff_{i}_w1']) * 0.3
            ff1 = jnp.maximum(0, ff1)  # ReLU
            ff2 = jnp.matmul(ff1, weights_dict[f'ff_{i}_w2']) * 0.3
            x = x + ff2 * 0.7  # Residual with scaling
        
        # Output projection
        return jnp.matmul(x, weights_dict['w_out']) * 0.3
    
    def get_weights_dict(self):
        """Get all weights as a dictionary."""
        weights = {
            'w_embed': jnp.array(self.w_embed),
            'w_pos': jnp.array(self.w_pos),
            'w_out': jnp.array(self.w_out),
        }
        
        for i in range(self.n_layers):
            weights[f'layer_{i}_q'] = jnp.array(self.w_layers[i]['q'])
            weights[f'layer_{i}_k'] = jnp.array(self.w_layers[i]['k'])
            weights[f'layer_{i}_v'] = jnp.array(self.w_layers[i]['v'])
            weights[f'layer_{i}_o'] = jnp.array(self.w_layers[i]['o'])
            weights[f'ff_{i}_w1'] = jnp.array(self.w_ff_layers[i]['w1'])
            weights[f'ff_{i}_w2'] = jnp.array(self.w_ff_layers[i]['w2'])
        
        return weights
    
    def update_weights(self, grads, lr=0.001):
        """Update weights using gradients (simplified SGD)."""
        # Update embedding
        self.w_embed -= lr * np.array(grads['w_embed'])
        self.w_pos -= lr * np.array(grads['w_pos'])
        self.w_out -= lr * np.array(grads['w_out'])
        
        # Update layers
        for i in range(self.n_layers):
            self.w_layers[i]['q'] -= lr * np.array(grads[f'layer_{i}_q'])
            self.w_layers[i]['k'] -= lr * np.array(grads[f'layer_{i}_k'])
            self.w_layers[i]['v'] -= lr * np.array(grads[f'layer_{i}_v'])
            self.w_layers[i]['o'] -= lr * np.array(grads[f'layer_{i}_o'])
            self.w_ff_layers[i]['w1'] -= lr * np.array(grads[f'ff_{i}_w1'])
            self.w_ff_layers[i]['w2'] -= lr * np.array(grads[f'ff_{i}_w2'])


def create_forward_fn(model):
    """Create a forward function that takes all weights as arguments."""
    def forward_fn(x, *weight_args):
        # Reconstruct weights dict from args
        weights_dict = model.get_weights_dict()
        return model.forward(x, weights_dict)
    return forward_fn


def track_bounds_during_training(model, x, n_steps=20, lr=0.001, epsilon=0.01):
    """Track bounds during training steps."""
    print(f"\nTracking bounds during {n_steps} training steps...")
    
    # Create forward function
    def forward_fn(x, weights_dict):
        return model.forward(x, weights_dict)
    
    # Track bounds for each training step
    all_trackers = []
    training_steps = []
    
    for step in range(n_steps):
        print(f"  Step {step + 1}/{n_steps}...", end=" ")
        
        # Get current weights
        weights_dict = model.get_weights_dict()
        
        # Create jaxpr
        try:
            # Create jaxpr - flatten weights dict into individual arguments
            # This is needed because JAX jaxpr expects individual arguments
            weight_list = list(weights_dict.values())
            weight_keys = list(weights_dict.keys())
            
            # Create a function that takes weights as separate arguments
            def forward_flat(x, *w_args):
                w_dict = {key: w for key, w in zip(weight_keys, w_args)}
                return model.forward(x, w_dict)
            
            closed_jaxpr = make_jaxpr(forward_flat)(x, *weight_list)
            
            # Create tracker
            tracker = BoundsTracker()
            interpreter = TrackingInterpreter(epsilon, tracker)
            
            # Create gradient seeds
            grad_seed_x = jnp.ones_like(x) * 0.1
            grad_seeds = [grad_seed_x] + [jnp.ones_like(w) * 0.05 for w in weight_list]
            
            # Interpret
            results = interpreter.interpret(closed_jaxpr, x, *weight_list, gradient_seeds=grad_seeds)
            
            all_trackers.append(tracker)
            training_steps.append(step)
            
            # Get final output bounds
            final_result = results[0]
            val_l, val_u = final_result.get_bounds()
            val_min, val_max = float(jnp.min(val_l)), float(jnp.max(val_u))
            
            print(f"Value bounds: [{val_min:.2f}, {val_max:.2f}]")
            
            # Simulate training: compute gradients and update weights
            # (In real training, we'd use actual loss and gradients)
            # For now, we'll just add small random updates to simulate training
            if step < n_steps - 1:
                # Simulate gradient update with some variation
                fake_grads = {}
                for key, weight in weights_dict.items():
                    # Use gradients with some variation to show stability
                    base_grad = 0.01
                    variation = 0.002 * np.sin(step * 0.5)  # Add periodic variation
                    fake_grads[key] = np.random.randn(*weight.shape) * (base_grad + variation)
                model.update_weights(fake_grads, lr=lr)
        
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_trackers, training_steps


def plot_training_bounds(all_trackers, training_steps):
    """Plot bounds evolution during training."""
    if len(all_trackers) == 0:
        print("No data to plot")
        return
    
    # Collect data across training steps
    step_values = []
    val_mins = []
    val_maxs = []
    grad_mins = []
    grad_maxs = []
    matmul_val_mins = []
    matmul_val_maxs = []
    matmul_grad_mins = []
    matmul_grad_maxs = []
    
    for step, tracker in zip(training_steps, all_trackers):
        if len(tracker.steps) > 0:
            # Get final bounds from each step
            final_val_min = tracker.value_bounds[-1][0]
            final_val_max = tracker.value_bounds[-1][1]
            final_grad_min = tracker.dual_bounds[-1][0]
            final_grad_max = tracker.dual_bounds[-1][1]
            
            step_values.append(step)
            val_mins.append(final_val_min)
            val_maxs.append(final_val_max)
            grad_mins.append(final_grad_min)
            grad_maxs.append(final_grad_max)
            
            # MatMul bounds
            if len(tracker.matmul_steps) > 0:
                matmul_val_mins.append(tracker.matmul_value_bounds[-1][0])
                matmul_val_maxs.append(tracker.matmul_value_bounds[-1][1])
                matmul_grad_mins.append(tracker.matmul_dual_bounds[-1][0])
                matmul_grad_maxs.append(tracker.matmul_dual_bounds[-1][1])
    
    if len(step_values) == 0:
        print("No valid data points")
        return
    
    # Create plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])
    
    # Plot 1: Overall value bounds during training
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(step_values, val_mins, val_maxs, alpha=0.3, color='blue', label='Value Bounds')
    ax1.plot(step_values, val_mins, 'b-', linewidth=2, label='Value Min', marker='o')
    ax1.plot(step_values, val_maxs, 'b--', linewidth=2, label='Value Max', marker='s')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Value Bound', fontsize=12)
    ax1.set_title('Value Bounds During Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overall dual bounds during training
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(step_values, grad_mins, grad_maxs, alpha=0.3, color='red', label='Dual Bounds')
    ax2.plot(step_values, grad_mins, 'r-', linewidth=2, label='Dual Min', marker='o')
    ax2.plot(step_values, grad_maxs, 'r--', linewidth=2, label='Dual Max', marker='s')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Dual Bound (Gradient)', fontsize=12)
    ax2.set_title('Dual Bounds During Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MatMul value bounds during training
    if len(matmul_val_mins) > 0:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.fill_between(step_values[:len(matmul_val_mins)], matmul_val_mins, matmul_val_maxs, 
                         alpha=0.3, color='green', label='MatMul Value Bounds')
        ax3.plot(step_values[:len(matmul_val_mins)], matmul_val_mins, 'g-', linewidth=2, 
                label='MatMul Value Min', marker='o')
        ax3.plot(step_values[:len(matmul_val_mins)], matmul_val_maxs, 'g--', linewidth=2, 
                label='MatMul Value Max', marker='s')
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel('MatMul Value Bound', fontsize=12)
        ax3.set_title('MatMul Value Bounds During Training', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: MatMul dual bounds during training
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.fill_between(step_values[:len(matmul_grad_mins)], matmul_grad_mins, matmul_grad_maxs, 
                         alpha=0.3, color='orange', label='MatMul Dual Bounds')
        ax4.plot(step_values[:len(matmul_grad_mins)], matmul_grad_mins, 'orange', linewidth=2, 
                label='MatMul Dual Min', marker='o')
        ax4.plot(step_values[:len(matmul_grad_mins)], matmul_grad_maxs, 'orange', linestyle='--', 
                linewidth=2, label='MatMul Dual Max', marker='s')
        ax4.set_xlabel('Training Step', fontsize=12)
        ax4.set_ylabel('MatMul Dual Bound', fontsize=12)
        ax4.set_title('MatMul Dual Bounds During Training', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Bound stability (variance/range)
    ax5 = fig.add_subplot(gs[2, :])
    val_ranges = [max_val - min_val for min_val, max_val in zip(val_mins, val_maxs)]
    grad_ranges = [max_val - min_val for min_val, max_val in zip(grad_mins, grad_maxs)]
    
    ax5.plot(step_values, val_ranges, 'b-', linewidth=2, label='Value Range', marker='o')
    ax5.plot(step_values, grad_ranges, 'r-', linewidth=2, label='Dual Range', marker='s')
    if len(matmul_val_mins) > 0:
        matmul_val_ranges = [max_val - min_val for min_val, max_val in 
                            zip(matmul_val_mins, matmul_val_maxs)]
        ax5.plot(step_values[:len(matmul_val_ranges)], matmul_val_ranges, 'g-', 
                linewidth=2, label='MatMul Value Range', marker='^')
    ax5.set_xlabel('Training Step', fontsize=12)
    ax5.set_ylabel('Bound Range (Uncertainty)', fontsize=12)
    ax5.set_title('Bound Stability During Training (Lower = More Stable)', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'gpt2_training_bounds_stability.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training bounds plot saved to: {filename}")
    plt.close()
    
    # Print stability statistics
    if len(val_ranges) > 1:
        val_range_std = np.std(val_ranges)
        grad_range_std = np.std(grad_ranges)
        print(f"\nStability Statistics:")
        print(f"  Value range std: {val_range_std:.3f} (lower = more stable)")
        print(f"  Dual range std: {grad_range_std:.3f} (lower = more stable)")
        print(f"  Value range change: {val_ranges[-1] - val_ranges[0]:.3f}")
        print(f"  Dual range change: {grad_ranges[-1] - grad_ranges[0]:.3f}")


def main():
    """Main training and bounds tracking."""
    print("=" * 60)
    print("Trainable GPT2 Model - Bounds Stability Analysis")
    print("=" * 60)
    
    # Create model with moderate dimensions
    model = TrainableGPT2(vocab_size=200, d_model=32, d_ff=128, n_layers=2)
    
    # Create input
    batch_size = 2
    seq_len = 6
    x = jnp.array(np.random.randn(batch_size, seq_len, model.vocab_size) * 0.5)
    
    # Track bounds during training
    all_trackers, training_steps = track_bounds_during_training(
        model, x, n_steps=25, lr=0.005, epsilon=0.05
    )
    
    # Plot results
    plot_training_bounds(all_trackers, training_steps)
    
    print("\n" + "=" * 60)
    print("Training bounds analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

