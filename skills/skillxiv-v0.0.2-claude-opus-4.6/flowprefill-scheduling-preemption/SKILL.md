---
name: flowprefill-scheduling-preemption
title: "FlowPrefill: Decoupling Preemption from Prefill Scheduling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.16603"
keywords: [LLM serving, scheduling, latency optimization, prefill, decoding, head-of-line blocking]
description: "Improve LLM serving under mixed workloads by decoupling execution granularity from scheduling frequency. Operator-level preemption allows fine-grained interruption at natural boundaries (attention, feed-forward layers) without efficiency loss. Event-driven scheduling triggers decisions only on request arrival/completion. Eliminates head-of-line blocking where long requests starve short time-sensitive ones. Achieves 5.6× higher goodput vs. baselines in production workloads."
---

# FlowPrefill: Fine-Grained Preemption Without Efficiency Loss

LLM serving systems face a fundamental trade-off: batch-oriented processing is efficient but creates long latencies for short requests waiting behind long-running ones (head-of-line blocking). Conversely, frequently pausing requests enables responsive preemption but fragments computation and wastes GPU cycles.

Traditional systems are forced to choose: either long fixed batch chunks (efficient but unresponsive) or short chunks (responsive but inefficient). FlowPrefill decouples these concepts: execute in large operator-granularity chunks for efficiency while scheduling (deciding which request runs next) at fine-grained intervals.

## Core Concept

FlowPrefill uses two complementary mechanisms:

**Operator-Level Preemption**: Pause execution at natural operator boundaries (completion of attention/feed-forward layers) rather than arbitrary fixed points. This allows interruption without wasting partial computations. Most operators align with layer boundaries, enabling efficient context switching.

**Event-Driven Scheduling**: Rather than checking for preemption at fixed frequencies, schedule only when requests arrive or complete. Between these events, execution continues without overhead, minimizing scheduling latency.

The combination allows responsive preemption (reacting to high-priority arrivals) with efficient computation (long execution runs), solving the efficiency-responsiveness trade-off.

## Architecture Overview

- **Operator Graph**: Represent model as DAG of operators (attention, linear, activation); nodes are individual operations
- **Preemption Points**: Identify safe boundaries between operators where execution can pause
- **Priority Queue**: Track waiting requests with priorities (e.g., token deadline, priority tier)
- **Event Listener**: Monitor request arrivals and completions
- **Scheduler**: On events, select next request to run; execute until next preemption point
- **Context Manager**: Save/restore execution state (activations, KV-cache) at preemption points
- **QoS Controller**: Enforce priority policies and latency targets through scheduling decisions

## Implementation

Identify preemption-safe boundaries in the model:

```python
def find_preemption_points(model):
    """
    Identify operator boundaries where execution can safely pause.
    Returns list of (layer_idx, operator_name, is_safe)
    """
    preemption_points = []

    for layer_idx, layer in enumerate(model.layers):
        # Attention operator
        if hasattr(layer, 'attention'):
            preemption_points.append((layer_idx, 'attention', True))

        # Feed-forward operator
        if hasattr(layer, 'mlp'):
            preemption_points.append((layer_idx, 'mlp', True))

    # Typically safe to preempt between layers (after layer norm)
    return preemption_points

def can_preempt_at_operator(model, operator_name):
    """
    Check if preemption at this operator is safe and efficient.
    Safe: doesn't leave partial computation hanging
    Efficient: doesn't duplicate work on resume
    """
    # Most layer-based operators are safe
    safe_operators = ['attention', 'mlp', 'norm', 'gelu']

    return operator_name in safe_operators
```

Implement operator-level execution with preemption:

```python
class PreemptibleOperatorExecutor:
    def __init__(self, model):
        self.model = model
        self.preemption_points = find_preemption_points(model)
        self.saved_activations = {}

    def execute_until_preemption(self, request_batch, next_preemption_point):
        """
        Execute model operators until reaching preemption point.
        Saves intermediate activations for resumption.
        """
        layer_idx, operator_name, _ = next_preemption_point

        # Load any saved activations from previous preemption
        if request_batch.request_id in self.saved_activations:
            activations = self.saved_activations[request_batch.request_id]
            x = activations['x']
            layer_start = activations['layer_idx']
        else:
            x = request_batch.input_ids
            layer_start = 0

        # Execute from layer_start to preemption point
        for l in range(layer_start, layer_idx + 1):
            layer = self.model.layers[l]

            if l < layer_idx:
                # Execute full layer
                x = layer(x)
            else:
                # Execute up to specific operator
                if operator_name == 'attention':
                    x = layer.attention(x)
                elif operator_name == 'mlp':
                    x = layer.mlp(x)

        # Save activation for resumption
        self.saved_activations[request_batch.request_id] = {
            'x': x,
            'layer_idx': layer_idx,
            'operator': operator_name
        }

        return x

    def resume_after_preemption(self, request_batch, current_point):
        """
        Resume execution after preemption from saved activation.
        """
        # Execution continues from where it was saved
        return self.execute_until_preemption(
            request_batch, current_point
        )
```

Implement event-driven scheduling:

```python
class EventDrivenScheduler:
    def __init__(self, model):
        self.model = model
        self.request_queue = []
        self.current_request = None
        self.executor = PreemptibleOperatorExecutor(model)
        self.next_preemption_point = 0

    def on_request_arrival(self, new_request):
        """Event handler: new request arrived."""
        self.request_queue.append(new_request)

        # Preempt current request if new one has higher priority
        if self.should_preempt(new_request):
            if self.current_request:
                self.preempt_current()
            self.schedule_next()

    def on_request_completion(self, request_id):
        """Event handler: request completed."""
        # Remove from queue
        self.request_queue = [r for r in self.request_queue if r.id != request_id]

        # Clean up saved state
        if request_id in self.executor.saved_activations:
            del self.executor.saved_activations[request_id]

        # Schedule next request
        self.schedule_next()

    def should_preempt(self, new_request):
        """Check if new request should preempt current."""
        if not self.current_request:
            return False

        # Priority-based: higher priority preempts lower
        if new_request.priority > self.current_request.priority:
            return True

        # Deadline-based: closer deadline preempts
        if new_request.deadline < self.current_request.deadline:
            return True

        return False

    def preempt_current(self):
        """Pause current request execution."""
        # Activations already saved by executor
        self.current_request = None

    def schedule_next(self):
        """Select next request to run."""
        if not self.request_queue:
            return

        # Sort by priority/deadline
        self.request_queue.sort(
            key=lambda r: (-r.priority, r.deadline)
        )

        self.current_request = self.request_queue[0]

    def execute_batch(self):
        """Main execution loop: run current request until preemption."""
        if not self.current_request:
            return

        # Execute until next preemption point
        preemption_point = self.executor.preemption_points[self.next_preemption_point]
        output = self.executor.execute_until_preemption(
            self.current_request, preemption_point
        )

        # Move to next preemption point for next run
        self.next_preemption_point += 1
        if self.next_preemption_point >= len(self.executor.preemption_points):
            # Request complete
            self.on_request_completion(self.current_request.request_id)
            self.next_preemption_point = 0
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Preemption granularity | Layer boundary | Finer grain (operator) if latency critical; coarser (block) if throughput critical |
| Priority levels | 2 (high/normal) | Increase (3–5 levels) for complex QoS policies |
| Batch size | 32 | Larger batches improve throughput but increase latency for stragglers |
| Scheduling frequency | On arrival | Can add timeout-based checks (e.g., check every 10ms) if needed |

**When to use**: For mixed-workload LLM serving with variable request sizes and latency requirements; real-time applications.

**When not to use**: For batch inference with homogeneous workload; scheduling overhead isn't justified.

**Common pitfalls**:
- Preemption at inefficient boundaries (e.g., mid-operator); stick to clean operator boundaries
- Not tracking saved activation memory; can exhaust GPU memory with many queued requests
- Aggressive priority preemption starving low-priority work; use aging or fairness mechanisms to prevent starvation

## Reference

FlowPrefill achieves 5.6× higher goodput in production workload evaluations by eliminating head-of-line blocking. The approach is compatible with existing LLM serving frameworks and requires only modest changes to execution control flow.
