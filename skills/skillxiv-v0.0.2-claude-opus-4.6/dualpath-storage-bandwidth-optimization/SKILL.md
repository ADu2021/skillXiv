---
name: dualpath-storage-bandwidth-optimization
title: "DualPath: Breaking Storage Bandwidth Bottleneck in Agentic LLM Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.21548"
keywords: [LLM inference, distributed serving, KV-cache optimization, bandwidth optimization, multi-turn workloads]
description: "Optimize disaggregated prefill-decoding LLM serving for multi-turn (agentic) workloads by introducing dual-path KV-cache loading. Traditional approach loads all KV-cache to prefill engines, saturating their storage network. DualPath loads to decoding engines first, then transfers via compute network (lower contention). Adaptive routing selects path based on real-time queue depths. Achieves 1.87× offline throughput and 1.96× online serving improvement."
---

# DualPath: Dual-Path KV-Cache Loading for Balanced Storage Network

Disaggregated LLM serving systems separate prefill and decoding operations onto different GPU engines for computational efficiency. However, this architecture creates an asymmetric bottleneck: massive KV-cache tensors must be loaded from storage to prefill engines before processing, saturating the prefill-side storage network. Meanwhile, decoding engines sit underutilized, with available but unused storage bandwidth.

Multi-turn agentic workloads exacerbate this problem: each turn requires loading new KV-caches (contexts from previous turns), creating repeated bandwidth spikes. Standard load balancing treats all storage requests identically, missing the opportunity to redistribute I/O across available resources.

## Core Concept

DualPath introduces two loading paths for KV-cache:

**Traditional Path**: Storage → Prefill Engines (direct, saturates storage NIC)

**Novel Path**: Storage → Decoding Engines → Prefill Engines (via compute network, distributes load)

The key insight is that compute network bandwidth (RDMA between engines) is abundant and usually underutilized, while storage network bandwidth is the bottleneck. By loading KV-cache to decoding engines first (lower contention), then transferring via compute network, the system distributes I/O load across more resources.

An adaptive controller selects which path to use based on real-time queue lengths and GPU utilization.

## Architecture Overview

- **Storage Network Monitor**: Track utilization of storage-to-prefill links; detect saturation
- **Compute Network Monitor**: Track compute network (RDMA) utilization; identify available capacity
- **Dual-Path Loader**: Support both loading paths with switch logic
- **Queue Depth Tracker**: Monitor storage queue lengths on each engine
- **Adaptive Selector**: Choose path based on current network state
- **QoS Controller**: Ensure latency SLAs despite load shifting
- **Cost Estimator**: Estimate completion time for each path; pick faster one

## Implementation

Implement storage path selection logic:

```python
class DualPathKVCacheLoader:
    def __init__(self, num_prefill_engines=4, num_decoding_engines=8):
        self.num_prefill_engines = num_prefill_engines
        self.num_decoding_engines = num_decoding_engines

        # Storage queue depths per engine
        self.prefill_queue_depths = [0] * num_prefill_engines
        self.decoding_queue_depths = [0] * num_decoding_engines

        # Network utilization
        self.storage_network_util = 0.0
        self.compute_network_util = 0.0

    def estimate_load_time(self, kv_cache_size_mb, path='traditional'):
        """
        Estimate loading time via specified path.
        path: 'traditional' or 'dual'
        Returns: estimated time in ms
        """
        if path == 'traditional':
            # Storage network bandwidth (typically 200 GB/s)
            bandwidth = 200.0 * 1000  # MB/s
            contention_factor = 1.0 + 0.3 * self.storage_network_util

            load_time = (kv_cache_size_mb / bandwidth) * contention_factor
            return load_time

        elif path == 'dual':
            # Two-phase load
            # Phase 1: Storage → Decoding (same bandwidth but less contention)
            phase1_bandwidth = 200.0 * 1000 * (1.0 - self.compute_network_util)
            phase1_time = (kv_cache_size_mb / phase1_bandwidth) * 0.5  # Less congestion

            # Phase 2: Decoding → Prefill via compute network (RDMA, much higher BW)
            phase2_bandwidth = 800.0 * 1000  # Compute network (4x faster)
            phase2_time = kv_cache_size_mb / phase2_bandwidth

            total_time = phase1_time + phase2_time
            return total_time

    def select_loading_path(self, kv_cache_size_mb, deadline_ms=None):
        """
        Select optimal loading path based on current network state.
        Returns: 'traditional' or 'dual'
        """
        time_traditional = self.estimate_load_time(kv_cache_size_mb, 'traditional')
        time_dual = self.estimate_load_time(kv_cache_size_mb, 'dual')

        if deadline_ms:
            # If either would exceed deadline, pick faster one
            if time_traditional > deadline_ms and time_dual <= deadline_ms:
                return 'dual'
            elif time_dual > deadline_ms and time_traditional <= deadline_ms:
                return 'traditional'

        # Pick faster path
        return 'dual' if time_dual < time_traditional else 'traditional'

    def load_kv_cache_traditional(self, kv_cache, prefill_engine_id):
        """
        Load KV-cache directly from storage to prefill engine.
        """
        # Update queue depth
        self.prefill_queue_depths[prefill_engine_id] += 1

        # Simulate loading time
        load_time = self.estimate_load_time(kv_cache.size_mb, 'traditional')
        self.storage_network_util += 0.1  # Increase utilization

        # Actual load (would be async in production)
        # transfer_to_engine(kv_cache, prefill_engine_id)

        # Decrement queue
        self.prefill_queue_depths[prefill_engine_id] -= 1
        self.storage_network_util -= 0.1

        return kv_cache

    def load_kv_cache_dual_path(self, kv_cache, prefill_engine_id):
        """
        Load KV-cache via dual path: Storage → Decoding → Prefill
        """
        # Select underutilized decoding engine
        least_loaded_decoding = min(
            range(self.num_decoding_engines),
            key=lambda x: self.decoding_queue_depths[x]
        )

        # Phase 1: Load to decoding engine
        self.decoding_queue_depths[least_loaded_decoding] += 1
        load_time_1 = self.estimate_load_time(kv_cache.size_mb, 'dual') / 2

        # Phase 2: Transfer from decoding to prefill via compute network
        self.prefill_queue_depths[prefill_engine_id] += 1
        load_time_2 = self.estimate_load_time(kv_cache.size_mb, 'dual') / 2

        self.decoding_queue_depths[least_loaded_decoding] -= 1
        self.prefill_queue_depths[prefill_engine_id] -= 1

        return kv_cache
```

Implement adaptive path selection with network monitoring:

```python
class AdaptiveKVCacheRouter:
    def __init__(self, loader):
        self.loader = loader
        self.monitoring_thread = None

    def update_network_utilization(self):
        """
        Periodically monitor storage and compute network utilization.
        In production, would read from actual network telemetry.
        """
        while True:
            # Query network interfaces
            storage_nic_util = get_storage_nic_utilization()  # 0.0-1.0
            compute_nic_util = get_compute_nic_utilization()  # 0.0-1.0

            self.loader.storage_network_util = storage_nic_util
            self.loader.compute_network_util = compute_nic_util

            time.sleep(0.1)  # Update every 100ms

    def route_kv_cache_load(self, kv_cache, prefill_engine_id, deadline_ms=None):
        """
        Intelligently route KV-cache load request.
        """
        # Select path
        path = self.loader.select_loading_path(kv_cache.size_mb, deadline_ms)

        if path == 'traditional':
            return self.loader.load_kv_cache_traditional(kv_cache, prefill_engine_id)
        else:
            return self.loader.load_kv_cache_dual_path(kv_cache, prefill_engine_id)
```

Integrate into serving system:

```python
class AgenticLLMServingWithDualPath:
    def __init__(self, num_prefill=4, num_decoding=8):
        self.loader = DualPathKVCacheLoader(num_prefill, num_decoding)
        self.router = AdaptiveKVCacheRouter(self.loader)

    def handle_prefill_request(self, request_batch, context_kv_cache):
        """
        Handle prefill request with dual-path KV-cache loading.
        """
        # Estimate deadline based on request priority
        deadline_ms = 100 if request_batch.priority == 'high' else 500

        # Select prefill engine (round-robin or queue-based)
        prefill_engine_id = 0

        # Load KV-cache adaptively
        kv_cache = self.router.route_kv_cache_load(
            context_kv_cache, prefill_engine_id, deadline_ms
        )

        # Proceed with prefill computation
        return prefill_engine_process(request_batch, kv_cache)
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Storage network BW | 200 GB/s | Measure actual NIC bandwidth for your hardware |
| Compute network BW | 800 GB/s | RDMA links typically 4–8× faster than storage |
| Utilization update interval | 100 ms | Faster updates for dynamic workloads; slower for stability |
| Path selection threshold | Auto | Adaptive selection based on real-time state |
| Min prefill queue depth | 2 | Queue buffering to mask load time variance |

**When to use**: For disaggregated LLM serving with separate prefill/decoding hardware, especially multi-turn (agentic) workloads with repeated KV-cache loads.

**When not to use**: For monolithic systems where prefill and decoding share hardware; dual path doesn't help if all operations are on same device.

**Common pitfalls**:
- Ignoring transfer overhead between engines; dual path has latency even with higher bandwidth
- Not accounting for KV-cache coherency; ensure data consistency across dual loading paths
- Assuming compute network is always available; check actual RDMA link availability before committing to dual path

## Reference

DualPath achieves 1.87× throughput improvement for offline inference and 1.96× for online serving on production workloads. The adaptive routing mechanism automatically balances load across available network resources, improving overall system efficiency without requiring code changes to existing services.
