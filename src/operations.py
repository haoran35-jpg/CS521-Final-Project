import jax.numpy as jnp


def interval_matmul(a_real_l, a_real_u, b_real_l, b_real_u):
    """
    Compute value bounds for matrix multiplication C = A @ B using Rump's algorithm 4.8.
    Reference: https://www.tuhh.de/ti3/paper/rump/Ru11a.pdf
    """
    # Midpoint-radius representation
    mA = (a_real_l + a_real_u) * 0.5
    rA = mA - a_real_l
    mB = (b_real_l + b_real_u) * 0.5
    rB = mB - b_real_l
    
    # Compute rhoA and rhoB
    sA = jnp.sign(mA)
    sB = jnp.sign(mB)
    absMA = jnp.abs(mA)
    absMB = jnp.abs(mB)
    rhoA = sA * jnp.minimum(absMA, rA)
    rhoB = sB * jnp.minimum(absMB, rB)
    
    # Compute radius and bounds
    rC = jnp.matmul(absMA, rB) + jnp.matmul(rA, (absMB + rB)) + jnp.matmul(-jnp.abs(rhoA), jnp.abs(rhoB))
    C1 = jnp.matmul(mA, mB) + jnp.matmul(rhoA, rhoB) - rC  # lower
    C2 = jnp.matmul(mA, mB) + jnp.matmul(rhoA, rhoB) + rC  # upper
    
    return C1, C2
