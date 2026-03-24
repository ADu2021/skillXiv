---
name: dash-faster-shampoo-optimizer
title: "DASH: Faster Shampoo via Batched Block Preconditioning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02016"
keywords: [Optimizer, Preconditioning, Matrix Operations, GPU Acceleration, Training Efficiency]
description: "Accelerate the Shampoo optimizer 4.8x using batched block-wise preconditioning and numerical approximations, enabling more frequent preconditioner updates without computational bottleneck."
---

# DASH: Faster Shampoo via Batched Block Preconditioning

## Problem Context

The Shampoo optimizer achieves superior training performance (won MLCommons AlgoPerf competition) but suffers from significant computational overhead. Computing inverse matrix roots—its primary bottleneck—scales as O(n³), forcing infrequent preconditioner updates that degrade optimization quality. This limits practical adoption despite strong theoretical properties.

## Core Concept

DASH introduces [batched block processing, numerical approximations, specialized GPU operations] to accelerate the most expensive components. By stacking preconditioner blocks into 3D tensors, DASH enables parallel GPU processing via batched operations, replacing sequential block computation with vectorized operations.

## Architecture Overview

- **Architectural optimization**: Stack blocks into 3D tensors for batched processing
- **Numerical improvements**: Newton-Denman-Beavers (NDB) iteration and Chebyshev polynomial approximations as alternatives to eigenvalue decomposition
- **Multi-Power-Iteration**: Optimal matrix scaling with faster convergence
- **Frequency boost**: Enable more frequent updates without wall-clock slowdown
- **Drop-in replacement**: Compatible with existing Shampoo implementations

## Implementation

### Step 1: Organize weight matrices into blocks

Partition parameter matrices into blocks and stack them for batch processing.

```python
# Block matrix organization
class BlockOrganizer:
    def __init__(self, block_size=256):
        self.block_size = block_size

    def partition_matrix(self, weight_matrix):
        """
        Partition weight matrix into blocks of size block_size x block_size.
        Returns list of blocks and metadata for reconstruction.
        """
        h, w = weight_matrix.shape
        blocks = []
        block_info = []

        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block_h = min(self.block_size, h - i)
                block_w = min(self.block_size, w - j)

                block = weight_matrix[i:i+block_h, j:j+block_w]
                blocks.append(block)

                block_info.append({
                    'row_start': i, 'row_end': i + block_h,
                    'col_start': j, 'col_end': j + block_w,
                    'shape': block.shape
                })

        return blocks, block_info

    def reconstruct_matrix(self, blocks, block_info, original_shape):
        """
        Reconstruct original matrix from blocks.
        """
        h, w = original_shape
        reconstructed = torch.zeros(h, w, device=blocks[0].device)

        for block, info in zip(blocks, block_info):
            rs, re = info['row_start'], info['row_end']
            cs, info['col_start'], info['col_end']
            reconstructed[rs:re, cs:ce] = block

        return reconstructed

    def batch_blocks(self, blocks, batch_size=32):
        """
        Organize blocks into batches for parallel processing.
        """
        batches = []
        for i in range(0, len(blocks), batch_size):
            batch = torch.stack(blocks[i:i+batch_size])
            batches.append(batch)

        return batches
```

### Step 2: Implement batched matrix root computation

Use batched GPU operations to compute inverse square roots in parallel.

```python
# Batched matrix inverse square root
def batched_matrix_inv_sqrt(matrices_batch, method='ndb', num_iterations=10):
    """
    Compute (M^T M)^{-1/2} for a batch of matrices.

    Args:
        matrices_batch: Tensor of shape (batch_size, n, n)
        method: 'ndb' (Newton-Denman-Beavers), 'eigen', or 'cheby'
        num_iterations: Iterations for iterative methods
    """
    batch_size = matrices_batch.shape[0]
    n = matrices_batch.shape[1]

    if method == 'ndb':
        # Newton-Denman-Beavers iteration
        # More stable and faster than eigenvalue decomposition
        Y = matrices_batch.clone()  # Numerator
        Z = torch.eye(n, device=matrices_batch.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )  # Denominator

        for _ in range(num_iterations):
            # NDB iteration
            Y_inv = torch.linalg.inv(Y)
            Z_inv = torch.linalg.inv(Z)

            Y_next = 0.5 * (Y + Z_inv)
            Z_next = 0.5 * (Z + Y_inv)

            Y = Y_next
            Z = Z_next

        inv_sqrt = Y  # Result

    elif method == 'cheby':
        # Chebyshev polynomial approximation
        # Fast for well-conditioned matrices
        # Compute eigenvalue bounds
        evals = torch.linalg.eigvalsh(matrices_batch)
        lambda_max = evals[..., -1]
        lambda_min = evals[..., 0]

        # Rescale to [-1, 1] interval
        center = (lambda_max + lambda_min) / 2.0
        half_width = (lambda_max - lambda_min) / 2.0

        # Chebyshev approximation: sum of Chebyshev polynomials
        inv_sqrt = torch.zeros_like(matrices_batch)

        for i in range(num_iterations):
            # Chebyshev polynomial evaluation (simplified)
            T_i = compute_chebyshev_polynomial(
                i, (matrices_batch - center) / half_width
            )
            inv_sqrt += T_i

    else:  # 'eigen'
        # Standard eigenvalue decomposition
        evals, evecs = torch.linalg.eigh(matrices_batch)
        inv_sqrt = evecs @ torch.diag_embed(1.0 / torch.sqrt(evals)) @ evecs.transpose(-2, -1)

    return inv_sqrt
```

### Step 3: Implement multi-power iteration for scaling

Optimize matrix scaling to balance numerical stability and convergence.

```python
# Multi-Power iteration for optimal scaling
def multi_power_iteration_scaling(matrix, num_iterations=5):
    """
    Compute optimal scaling for matrix using power iteration.
    This stabilizes subsequent root computations.
    """
    # Initialize random vector
    v = torch.randn(matrix.shape[0], 1, device=matrix.device)
    v = v / torch.norm(v)

    # Power iteration
    for _ in range(num_iterations):
        v = matrix @ v
        v = v / torch.norm(v)

    # Estimate largest eigenvalue via Rayleigh quotient
    lambda_max = (v.T @ matrix @ v) / (v.T @ v)

    # Scaling factor: normalize largest eigenvalue to 1
    scaling = 1.0 / (lambda_max + 1e-8)

    return scaling, lambda_max.item()
```

### Step 4: Integrate into optimizer step

Create a drop-in replacement for Shampoo that uses DASH acceleration.

```python
# DASH optimizer
class DASHShampoo(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        eps=1e-10,
        block_size=256,
        update_freq=1,
        matrix_root_method='ndb'
    ):
        defaults = dict(
            lr=lr, eps=eps, block_size=block_size,
            update_freq=update_freq, matrix_root_method=matrix_root_method
        )
        super().__init__(params, defaults)

        self.block_organizer = BlockOrganizer(block_size=block_size)
        self.step_count = 0

    def step(self, closure=None):
        """
        Single optimization step using DASH-accelerated Shampoo.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['H'] = torch.eye(
                        grad.shape[1], device=grad.device
                    ) if len(grad.shape) == 2 else grad.numel()

                state['step'] += 1

                # Update preconditioning matrix H = grad @ grad^T
                if len(grad.shape) == 2:
                    # Matrix parameter
                    grad_norm = grad / (torch.norm(grad) + group['eps'])
                    state['H'] += grad_norm @ grad_norm.T

                    # Block-wise inverse square root computation
                    if state['step'] % group['update_freq'] == 0:
                        blocks, block_info = self.block_organizer.partition_matrix(
                            state['H']
                        )
                        batch_blocks = self.block_organizer.batch_blocks(
                            blocks, batch_size=32
                        )

                        # Batched computation
                        inv_sqrt_blocks = []
                        for batch in batch_blocks:
                            inv_sqrt_batch = batched_matrix_inv_sqrt(
                                batch, method=group['matrix_root_method']
                            )
                            inv_sqrt_blocks.extend(inv_sqrt_batch)

                        # Reconstruct and apply preconditioned update
                        H_inv_sqrt = self.block_organizer.reconstruct_matrix(
                            inv_sqrt_blocks, block_info, state['H'].shape
                        )

                        # Parameter update
                        p.data -= group['lr'] * (grad @ H_inv_sqrt)

                else:
                    # Vector parameter: use diagonal approximation
                    state['H'] += grad ** 2
                    h_inv_sqrt = 1.0 / torch.sqrt(state['H'] + group['eps'])
                    p.data -= group['lr'] * grad * h_inv_sqrt

        return loss
```

### Step 5: Benchmark and validate

Compare DASH against standard Shampoo to verify speedup and convergence.

```python
# Benchmarking utility
def benchmark_optimizer(
    model, train_loader, optimizer_class, optimizer_kwargs,
    num_epochs=5, device='cuda'
):
    """
    Benchmark optimizer training speed and convergence.
    """
    import time

    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.CrossEntropyLoss()

    wall_times = []
    losses = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - epoch_start
        wall_times.append(epoch_time)
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

    return {
        'wall_times': wall_times,
        'losses': losses,
        'total_time': sum(wall_times)
    }
```

## Practical Guidance

**When to use**: Large-scale model training where preconditioner computation is a bottleneck (1B+ parameters). Most beneficial for dense, fully-connected layers.

**Hyperparameters**:
- **Block size**: 256 (typical); balance between parallelism and computation per block
- **Update frequency**: 1 (update preconditioner every step); increase to 2-5 for larger savings
- **Matrix root method**: 'ndb' (recommended default for stability), 'cheby' (fast for well-conditioned)
- **Learning rate**: Same as standard Shampoo; no tuning needed

**Key performance metrics**:
- Speedup: 4-4.83x on standard Shampoo implementation
- Wall-clock improvement: ~40-50% overhead vs. SGD (vs. 90%+ for unoptimized Shampoo)
- Convergence: Often better than SGD due to improved preconditioner estimation

**Common pitfalls**:
- Block size too small → excessive overhead from block management
- Block size too large → reduces parallelism
- Forgetting to use batched operations → negates acceleration benefits
- Not validating numerical stability with NDB; eigenvalue decomposition safer but slower

**Scaling**: Benefits scale with parameter count. Minimal benefits for small models (<100M). Optimal for dense 1B-70B models.

## Reference

Paper: https://arxiv.org/abs/2602.02016
Code: Available at author's repository
Related work: Shampoo optimizer, preconditioning, second-order optimization
Benchmarks: Llama-953M, perplexity metrics, training curves
