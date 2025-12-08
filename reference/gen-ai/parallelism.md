# Parallelism in GenAI Models

*Single-Node Training*

When the model and the data are processed on a single device.

GPT-3 would take over [350 years](https://lambdalabs.com/blog/demystifying-gpt-3#6) to train on a single NVIDIA V100 GPU.

## Distributed ML

### Challenges

- Communication overhead
- Synchronization challenges
- Fault tolerance

Ensuring all nodes have the latest model parameters and updates requires careful coordination.

### Data Parallelism

Partition the dataset and distribute the subsets to multiple GPUs containing the same model.

There is a final sync of model/gradient updates.

GPT-3 can be [trained in a month](https://arxiv.org/pdf/2104.04473) using 1,024 NVIDIA A100 GPUs. 

*Parameter server*

This technique uses a separate server to manage the model’s weights.

It aggregates and updates the model weights by pulling individual gradients from each training server.

This approach is centralized, with a single point of failure.

*Peer-to-peer synchronization (P2P Sync)*

Decentralized approach where each server communicates with its peers to share updates.

Techniques:

| Name       | Description       | Challenges      |
| ---------- | ----------------- | --------------- |
| AllReduce  | Each node contributes its local gradients to a global average | Communication complexity adds up |
| Ring AllReduce | Ring topology - each worker communicates with 2 neighbors | Slower sequential info passing |
| Hierarchical AllReduce | Clusters of servers run AllReduce, with coordinator servers sharing aggregating results per cluster | Complex/expensive |

### Model Parallelism

Splits the model across multiple servers.

Used to train models that can't fit on one device. Also used for inferences.

Methods of model partitioning:

- *Layer-wise partitioning:* divide layers. For example, the input, hidden, and output layers could be placed on separate GPUs in a neural network. There can be bottlenecks if layers have strong dependencies.
- *Operator-wise partitioning:* split individual operations to different GPUs. For example, a matrix multiplication operation within a layer could be split across several GPUs. This requires careful synchronization.

### Hybrid parallelism

Data and model parallelism combined.

The dataset is split across nodes (data parallelism), and the model within each node is further split across GPUs (model parallelism).

Mainly used for extremely large models that don't fit on one GPU.

## Handling Parallelism Challenges

### Fault tolerance

In large distributed systems, the risk of node failure or communication errors increases.

Fixes:

- *Checkpointing:* Save intermediate states periodically to recover from failures.
- *Redundancy:* Use backup workers or mirrored model replicas to handle failures.
- *Monitoring:* Set up monitoring among the servers to ensure any error is reported and handled gracefully.

There are tradeoffs between allocating resources to training servers, and allocating them to replication.

### Hardware Heterogeneity

Difference devices/GPUs can have different memory or computing resources.

Fixes:

- Device-specific workloads
- Homogenous hardware clusters for specific tasks/workloads (e.g. the same class of GPUs)

The NVIDIA H100 performs 6.4x faster than the A100, so we might assign those GPUs 6.4x the work to do.

### Load imbalance

If certain GPUs handle more work than others, this results in idle time for some devices.

Fixes:

- *Dynamic work allocation:* adapt workload distribution based on each GPU’s computational capacity
- *Partition optimization:* assign layers according to computational capacity



























