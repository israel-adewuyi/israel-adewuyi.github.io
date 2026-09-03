---
layout: distill
auto_resize_iframes: true
ledger_published: true
title: "Three axes of Parallelism"
description: "Composing DP, PP and EP"
date: 2026-09-03
topic: "Distributed training"

# Optional archive image. The path is relative to assets/img/.
# image: "folder/hero-image.png"
# caption: "A short description of the archive image."

# Optional: place the matching .bib file under _bibliography/.
# bibliography: references.bib
---

## tl;dr

- [nanoTitan](https://github.com/israel-adewuyi/nanoTitan) is a distributed training stack, composing DP, PP and EP.
- We give a deep dive into the details of each of these parallelism strategies and their composition, as well as other surrounding details like gradient clipping.
- Below is a summary of the results across experiments on 2, 4 and 8 GPUs. **Experiments under the same number of gpus** processes the same number of tokens / step, to keep comparison relatively principled.
<figure id="fig-parallelism-results-explorer" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#overview"
    title="Interactive comparison of nanoTitan parallelism results by metric and GPU count"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(820px, calc(1150px - 45vw), 1000px); border: 0;"
  ></iframe>
  <figcaption>
    Metrics Summary
  </figcaption>
</figure>


## Introduction

Training larger models on increasing amounts of data requires training on more GPUs but each GPU added to the fleet adds some communication cost and each strategy presents its own unique set of challenges and tradeoffs to consider.

We explore 3 parallelism strategies (Data Parallelism, Pipeline Parallelism and Expert Parallelism), first individually, then their (2D/3D) compositions. We also examine their communication patterns, discuss the bottleneck they alleviate and then discuss some implementation details from [nanoTitan](https://github.com/israel-adewuyi/nanoTitan) and report measurements of throughput, memory usage, communication overhead, and scaling behaviour.

- All experiments in this report are single-node experiments, with max of 8 GPUs. Experiments on 1, 2 and 4 GPUs were conducted on RTX 3090 and experiment on 8 GPUs, on 3060.
- We ran each experiment under 2/4 GPUs for 200 steps and reported metrics are averaged over step 50 to 200. For the 8-GPUs experiment, we ran for 100 steps and the reported metrics are averaged over step 50 to 100.
- PyTorch's distributed primitives are used for the nanoTitan implementation and the exact implementation of the communication libraries are out of the scope of this report.

### Notation and accounting conventions

The same notation is used across the DP, PP, EP, and composition sections.

| Symbol | Meaning |
|---|---|
| $B$ | Global batch size |
| $b$ | batch size per data rank |
| $M$ | Number of pipeline microbatches |
| $m$ | batch size of one pipeline microbatch|
| $S$ | Sequence length |
| $D$ | Hidden dimension, `d_model` |
| $L$ | Number of layers |
| $P_{DP}$ | Data-parallel size |
| $P_{PP}$ | Pipeline-parallel size |
| $P_{EP}$ | Expert-parallel size |
| $E$ | Total number of experts |
<!-- | $W$ | Effective link bandwidth (bytes/sec) | -->


<details>
  <summary>See model architecture details</summary>
- Decoder-only MoE model<br>
- MHA<br>
- Pre-norm layers<br>
- ROPE for positional encoding<br>
  <table>
    <tr><th>Layers</th><td>16</td></tr>
    <tr><th>Hidden dimension</th><td>512</td></tr>
    <tr><th>Attention heads</th><td>8</td></tr>
    <tr><th>Experts</th><td>20 ($K = 2$)</td></tr>
  </table>
</details>

### Background
- A **rank** is an ID for a process in distributed training. A process controls a GPU and we use GPU and rank interchangeably.
- A **data rank** is a rank that receives a distinct shard of the input batch. In nanoTitan, DP and EP read in independent input shards, whereas PP sends on an existing shard through multiple pipeline stages. Therefore, $B=P_{\mathrm{DP}}\cdot P_{\text{EP}} \cdot b$ and $b = M \cdot m$.
- A **Process Group** defines a set of ranks that are communicating with each other. A rank can be in multiple process groups, which likely means its in comms with multiple sets of GPUs.
- The communication patterns used and reported here are:
    - [**AllReduce:**](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) Aggregates a tensor across all the ranks in the process group and returns the aggregate to all the ranks.
    - [**All-to-All:**](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.all_to_all_single) Exchanges different pieces of a tensor between ranks in the process group, such that every rank is both a sender and a receiver, sending a chunk and receiving a chunk.
    - [**Send/recv:**](https://docs.pytorch.org/docs/2.13/distributed.html#point-to-point-communication) Moves a tensor between two ranks.

## Data Parallel (DP)
When the model parameters, optimizer state, gradients and activations fit on a GPU, DP is the simplest way to scale training. In DP, parameters, optimizer states, and gradient buffers are replicated across a group of ranks, a DP group, but the input batch and consequently the activations are sharded along the batch dimension. Each replica gets an input batch of size $\left[b, \, S\right]$ where $b = \frac{B}{P_{DP}}$.

The batch size, B and consequently, throughput (measured in tokens/sec) scales with the DP group size <d-footnote>This implies keeping `per-data-rank-batch-size` b fixed as we increase $P_\text{DP}$, so $B = P_\text{DP}b$ grows with more GPUs added to the DP group. </d-footnote>.

<figure id="fig-dp" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/dp-draft-a-mirror.html' | relative_url }}"
    title="Interactive data-parallel forward and backward passes"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(200px, 60vw, 420px); border: 0;"
  ></iframe>
  <figcaption>
    DP Overview
  </figcaption>
</figure>

During the forward pass, there is no communication between the replicas in the DP group <d-footnote>Except for the state_dict() broadcast at the start of the training.</d-footnote>. However, during the backward pass, each rank computes the gradients for its parameter tensors and the gradients are averaged across all the ranks([Eq. 1](#dp-allreduce)), using **AllReduce**. The averaged gradients, equal on every rank, are then used to take the optimizer step. An added bonus of the optimizer step is that (ideally, within some bounds), the model parameters across the DP group would be the same after the optimizer step.

<figure id="dp-allreduce" class="l-body" markdown="1">
$$
\nabla G_B
=
\frac{1}{P_{DP}}
\sum_{r=0}^{P_{DP}-1}\nabla G_r.
$$

<figcaption style="text-transform: none;">
  <strong>Equation 1.</strong> The global gradient is the average of the gradients computed by the DP replicas.
</figcaption>
</figure>

For a broader introduction to data parallelism, see [Siboehm's DP explainer](https://siboehm.com/articles/22/data-parallel-training)

### Implementation Details
#### Distributed DataLoader

Each DP replica gets a different portion of the training data. In nanoTitan, we implemented it as follows;
- The dataset is streamed from Huggingface and shuffled using fixed seed across DP group.
- `[split_dataset_by_node](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.distributed.split_dataset_by_node)` is called on the shuffled stream, passing in the rank's `data_rank` and `world_size` which does the actual partitioning across ranks.
- Each resulting stream is tokenized and packed independently into sequences of length $S$.
- The [DataLoader](https://docs.pytorch.org/docs/2.13/data.html#torch.utils.data.DataLoader) then batches these sequences using the per-device batch size $b$.


<details>
  <summary>Code: Dataloading Snippet</summary>

  <pre><code class="language-python">
class PackedTokenDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        split: str = "train",
        shuffle: bool = True,
    ):
        self.seq_len = seq_len
        dataset = load_dataset(name, split=split, streaming=True)
        if shuffle:
            dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
        self.dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
        ...

def prepare_trainloader(train_dataset: PackedTokenDataset):
  train_loader = DataLoader(
      train_dataset,
      batch_size=per_device_batch_size,
      shuffle=False,
      ...
  )
  return train_loader
  </code></pre>
</details>

#### Reducer

Launching AllReduce for every parameter tensor is inefficient as each launch has a fixed cost and too many launches increases latency. Launching the AllReduce only after the backward pass is also inefficient as it could be faster to overlap AllReduce of computed gradients with gradient computation of other parameters <d-footnote>Readers are referred to the fantastic blog post by [Siboehm](https://siboehm.com/articles/22/data-parallel-training), again!</d-footnote>.

The implemented solution is to group gradients into buckets;

- Assign parameters to buckets in approximately reverse order of `model.parameters()`.
<details>
  <summary>Code: Bucketting snippet</summary>
  <pre><code class="language-python">
buckets = []
current_bucket = None

for param in reversed(model.parameters()):
    param_bytes = param.numel() * param.element_size()

    if (current_bucket is None or current_bucket["size_bytes"] + param_bytes > bucket_size):
        current_bucket = {"params": [], "size_bytes": 0}
        buckets.append(current_bucket)

    current_bucket["params"].append(param)
    current_bucket["size_bytes"] += param_bytes
  </code></pre>
</details>
- Register an autograd hook for each parameter. The hook function passed in, is where we launch the AllReduce.
- When a gradient for a parameter tensor is produced, mark its position in the bucket ready.
- Once all the parameters for a bucket are marked as ready, launch its AllReduce asynchronously (this allows backward to continue simultaneously). Importantly, all ranks must launch AllReduce in the same order.
- Continue computing gradients for earlier layers while the collective runs.
- Before the optimizer step, wait for every outstanding collective and expose the averaged gradients to the parameters.


Bucket size controls the balance between latency and overlap. At the lower end, each parameter is assigned to its own bucket and in this case, buckets become ready earlier and provide more opportunity to hide communication, but
they launch more collectives. On the other hand, larger buckets pay less launch overhead in total and tend to use bandwidth more efficiently, but become ready later and leave a smaller overlap window.

<figure id="fig-bucket-allreduce" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/bucket-size-vs-all-reduce-calls.html' | relative_url }}"
    title="Bucket size versus number of AllReduce calls"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(320px, 55vw, 350px); border: 0;"
  ></iframe>
  <figcaption>
    Bucket sizes vs Number of AllReduce launches
  </figcaption>
</figure>


#### Gradient clipping

Clipping gradient scales the magnitude of the gradient, while keeping its direction intact. Given a tensor of gradients represented as `grad` with `total_norm`, the formula for clipping the gradient is

$$
grad = grad \cdot \min \left(\frac{\text{max_norm}}{\text{total_norm} + 1e-6}, 1\right)
$$

We clip the gradients independently on each rank, after AllReduce has been performed. The gradients, post-AllReduce and post-clipping are identical across the DP group.


### Experimental Results

<figure id="fig-dp-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#dp"
    title="Data-parallel experiment results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    DP results
  </figcaption>
</figure>



## Pipeline Parallel (PP)
DP makes the assumption that the model fits on a device and while increasing the number of replicas increases the throughput in token/sec, it doesn't reduce the model states stored on each GPU. Once the model states no longer fit on one device, we have to partition the model itself.

PP shards the model across its layers and places each group of layers on separate ranks, each called a **pipeline stage**, with the ranks forming a `PP group`. With $P_{PP}$ pipeline stages, a model with $L$ layers is divided such that there are $\frac{L}{P_{PP}}$ layers assigned to each stage. A stage only has access to its own local parameters, which it uses in the forward/backward pass, before passing the intermediate activations/derivatives to the next stage<d-footnote>Caveat to this is with tied-embedding. Stage 0 and $P_{PP} - 1$ might be using the same embedding head, just transposed.</d-footnote>.

<figure id="fig-pp" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/pp-draft-a-layer-split.html' | relative_url }}"
    title="Interactive pipeline-parallel layer split"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(200px, 60vw, 420px); border: 0;"
  ></iframe>
  <figcaption>
    PP Overview
  </figcaption>
</figure>

For a microbatch of size $m$, stage $i$ sends activations tensors of shape $\left[m,\, S,\, D\right]$ to stage $i + 1$ and during the backward pass, stage $i$ sends gradients w.r.t its input activations of the same shape to stage $i - 1$.

PP implemented naively makes poor use of the GPUs because later stages are idle while earlier stages are performing computation and earlier stages are also idle while later stages are performing computation. There are different pipeline schedules that have been developed over the years to combat these *bubbles*, see [Siboehm's pipeline-parallelism explainer](https://siboehm.com/articles/22/pipeline-parallel-training) for a broader explanation on pipeline schedules. Other interesting schedules are DualPipe from Deepseek, PipeDream, PipeDream-2BW.


### Implementation Details

With the entire model on one rank, a forward pass call, `model(x)`, implicitly computation goes from the embedding matrix through the layers and then the LM head. After sharding in PP, this breaks down because only stage 0 has the embedding matrix and only stage $P_{PP} - 1$ has the LM head, with intermediate ranks owning a couple of layers and receiving activations, not tokens, as their input. The right mental model here is each stage owns a sequence of modules, gets a tensor, runs forward pass and returns a hidden-state tensor.

#### Build the stage-local model

Before training, each PP rank is assigned a stage index and a consecutive range of transformer layers. The rank constructs only the modules it owns and the optimizer tracks states for only the local parameters. Each stage also records whether it's the first or last stage and it records the ranks of its previous and next neighbours.

<details>
    <summary>Code: Stage construction snippet</summary>
    <pre><code class="language-python">
def get_layer_bounds(cfg: Config, pp_rank: int):
    per_rank_layers = cfg.model.n_layers // cfg.runtime.pp_size
    start_idx = pp_rank * per_rank_layers
    end_idx = (pp_rank + 1) * per_rank_layers
    return (start_idx, end_idx)

def get_model_shard_specs(dim: ParallelDims, cfg: Config):
    has_token_embed = dim.is_pp_first_stage
    has_pos_embed = dim.is_pp_first_stage
    layer_start, layer_end = get_layer_bounds(cfg, dim.pp_rank)
    ...

    spec = ModelShardSpec(
        has_token_embed=has_token_embed,
        has_pos_embed=has_pos_embed,
        layer_start=layer_start,
        layer_end=layer_end,
        ...
    )

    return spec  
    </code></pre>
</details>

So on each stage, running `model(x)` runs the forward pass computation on the parameters hosted on the stage.

#### Communications across ranks.

`torch.distributed.send` and `torch.distributed.recv` are used for communication between stages. The receiver must first allocate a buffer with the expected shape $\left[m,\, S,\, D\right]$ and dtype and then receive into the buffer.

<details>
  <summary>Code: forward and backward boundary transport</summary>
  <pre><code class="language-python">
def recv_forward(self, microbatch_id):
    stage_input = self._create_activation_buffer()
    dist.recv(stage_input, src=self.dim.prev_pp_rank, group=self.dim.pp_group)
    return stage_input

def send_forward(self, microbatch_id, stage_output):
    dist.send(stage_output, dst=self.dim.next_pp_rank, group=self.dim.pp_group)

def recv_backward(self, microbatch_id):
    output_grad = self._create_activation_buffer()
    dist.recv(output_grad, src=self.dim.next_pp_rank, group=self.dim.pp_group)
    return output_grad

def send_backward(self, microbatch_id, input_grad):
    dist.send(input_grad, dst=self.dim.prev_pp_rank, group=self.dim.pp_group)
  </code></pre>
</details>

<br>

#### GPipe Schedule

The scheduler first divides the per-data-rank batch of size $b$ into $M$ microbatches of size $m$. In the forward pass, the first stage reads tokens as the stage input while every later stage receives an activation as the stage input. Each non-final stage sends its result onward and the final stage computes the loss. The backward pass visits the microbatches in reverse. The final stage starts from its loss and calls `loss.backward()` while every earlier stage receives an output-activation gradient, runs local backward, and sends its input-activation gradient to the preceding rank.

<details>
  <summary>Code: GPipe training loop snippet</summary>
  <pre><code class="language-python">for microbatch_id, (x, y) in enumerate(
    zip(microbatch_x, microbatch_y, strict=False)
):
    stage_input = (
        x.to(pipeline.device)
        if is_pp_first_stage
        else pipeline.recv_forward(microbatch_id)
    )
    stage_output = pipeline.forward_microbatch(
        microbatch_id,
        model,
        stage_input,
        y,
    )
    if not is_pp_last_stage:
        pipeline.send_forward(microbatch_id, stage_output)


for microbatch_id in reversed(range(len(microbatch_x))):
    output_grad = (
        None
        if is_pp_last_stage
        else pipeline.recv_backward(microbatch_id)
    )
    input_grad = pipeline.backward_microbatch(
        microbatch_id,
        output_grad,
        sync_gradients=microbatch_id == 0,
    )
    if not is_pp_first_stage:
        pipeline.send_backward(microbatch_id, input_grad)</code></pre>
</details>

#### 1F1B schedule

We still divide the input batch into $M$ microbatches, but `1F1B` begins the backward pass as soon as the gradients are ready. Once the pipeline is full, a stage alternates between computing a forward and a backward pass, hence the name **1F1B**. Each backward pass relaxes the activation memory pressure of the corresponding microbatch and this, in conjunction with the activation checkpointing, makes `1F1B` a more memory efficient schedule than `GPipe`.

We implement a training step in three phases:
  - **Warmup phase**: Each stage takes its stage input, runs forward pass and passes the activations to the next stage. Earlier stages have a longer warmup phase because earlier microbatches have to be processed on all stages before backward computation with respect to that microbatch begins. For pipeline stage $p$, it has $\min(M, P_{PP}-p)$ steps.
  - **Steady phase**: Once the first activation gradient is ready, each stage now runs a backward pass followed by a forward pass. Forward pass corresponds to the next microbatch in the pipeline and backward pass corresponds to the gradient with respect to the most outstanding microbatch.
  - **Cooldown phase**: Each stage completes its backward pass and at this point, no more microbatches to be processed in the pipeline.

We maintain an index for both the microbatch forward pass and backward pass; `fwd_idx` points to the next microbatch to be fed into the pipeline and `bwd_idx` points to the oldest microbatch whose gradient hasn't been computed yet. This is important because unlike the GPipe implementation where the backward pass is run in reverse order of microbatch indices, `1F1B` runs backward pass in increasing order and gradient sync for the step occurs at `bwd_idx == num_microbatches - 1`.

In the warmup and cooldown phase, we use `send` and `recv` from the torch.distributed library and during the steady phase, we use `dist.batch_isend_irecv`. It allows two ranks to perform a point-to-point communication i.e rank i sends forward to rank i + 1 and rank i + 1 sends backward to rank i, while also avoiding deadlock that blocking point-to-point calls can introduce.

<details>
  <summary>Code: 1F1B training loop</summary>
  <pre><code class="language-python">def run_1F1B(pipeline, model, microbatch_x, microbatch_y):
    M = len(microbatch_x)
    warmup_steps = min(M, pipeline.dim.pp_size - pipeline.dim.pp_rank)
    fwd_idx, bwd_idx = 0, 0

    # Warmup: run forwards until the first gradient can reach this stage.
    while fwd_idx &lt; warmup_steps:
        stage_input = (
            microbatch_x[fwd_idx]
            if pipeline.dim.is_pp_first_stage
            else pipeline.recv_forward(fwd_idx)
        )
        stage_output = pipeline.forward_microbatch(
            fwd_idx, model, stage_input, microbatch_y[fwd_idx]
        )
        if not pipeline.dim.is_pp_last_stage and fwd_idx &lt; warmup_steps - 1:
            pipeline.send_forward(fwd_idx, stage_output)
        fwd_idx += 1

    # Steady state: one backward, followed by one forward.
    while True:
        output_grad = (
            None
            if pipeline.dim.is_pp_last_stage
            else pipeline.send_forward_recv_backward(stage_output)
        )
        input_grad = pipeline.backward_microbatch(bwd_idx, output_grad)
        bwd_idx += 1

        if fwd_idx == M:
            if not pipeline.dim.is_pp_first_stage:
                pipeline.send_backward(bwd_idx - 1, input_grad)
            break

        stage_input = (
            microbatch_x[fwd_idx]
            if pipeline.dim.is_pp_first_stage
            else pipeline.recv_forward_send_backward(input_grad)
        )
        stage_output = pipeline.forward_microbatch(
            fwd_idx, model, stage_input, microbatch_y[fwd_idx]
        )
        fwd_idx += 1

    # Cooldown: drain the remaining backward passes.
    while bwd_idx &lt; M:
        output_grad = (
            None
            if pipeline.dim.is_pp_last_stage
            else pipeline.recv_backward(bwd_idx)
        )
        input_grad = pipeline.backward_microbatch(bwd_idx, output_grad)
        if not pipeline.dim.is_pp_first_stage:
            pipeline.send_backward(bwd_idx, input_grad)
        bwd_idx += 1</code></pre>
</details>

#### Gradient Clipping

In PP, the entire model no longer resides on a single GPU. To recover the same norm as single-device training, each stage computes its local **squared** gradient norm which is summed across the PP group (using AllReduce), and the square root is taken once. Every stage then scales its local gradients by the same value.

Let $\Theta_s$ be the parameters owned by pipeline stage $s$, and let $g_p = \nabla_p\mathcal{L}$. Because the stage partitions are disjoint,

$$
\begin{aligned}
\lVert g \rVert_2^2
&= \sum_{p\in\Theta}\lVert g_p\rVert_2^2 \\
&= \sum_{s=0}^{P_{PP}-1}\sum_{p\in\Theta_s}\lVert g_p\rVert_2^2 \\
&= \sum_{s=0}^{P_{PP}-1}\lVert g_s\rVert_2^2.
\end{aligned}
$$

Therefore, each stage can recover the full-model gradient norm using

$$
\lVert g\rVert_2
=
\sqrt{
\operatorname{AllReduce}_{\mathrm{SUM}}
\left(\lVert g_s\rVert_2^2\right)
}.
$$

### Experimental Results

<figure id="fig-pp-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#pp"
    title="Pipeline-parallel experiment results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    PP results
  </figcaption>
</figure>

## Expert Parallel (EP)
A Mixture of Experts (MoE) layer is a nice way to increase model capacity by replacing a single MLP with $E$ expert MLPs while keeping computation roughly constant by only activating the top-$K$ experts per token. The downside is that the expert parameters, gradients, and optimizer states have to be kept in memory. At some $E$, along with other model parameters, it becomes impractical to fit them on a single GPU.

Across $P_{EP}$ ranks in the Expert Parallel group, Expert parallelism 
- shards $E$ experts, so each rank owns $E_{\mathrm{local}}=\frac{E}{P_{EP}}$ experts <d-footnote>This assumes $E$ is divisible by $P_{EP}$</d-footnote>.
- replicates the non-expert weights, including the router.
- shards the input data, so each rank in the EP group processes a different, disjoint set of tokens.

The router on each rank still scores tokens against the global pool of experts $E$ and not $E_{\mathrm{local}}$, so tokens on each rank can select experts across all ranks.

<!-- Top-2 routing across two EP ranks. Each token keeps the same color as it is dispatched to local and remote experts, returned to its source rank, and differentiated during backward. -->

<figure id="fig-ep-top2-routing" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/ep-top2-mirror.html' | relative_url }}"
    title="Top-2 expert routing across two expert-parallel ranks"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(430px, 100vw, 700px); border: 0;"
  ></iframe>
  <figcaption>
    EP Overview
  </figcaption>
</figure>

### Implementation

#### Mapping experts to EP ranks

We extend the [stage-local model from PP](#build-the-stage-local-model) by allowing each rank load its own subset of experts such that rank i holds $[i \cdot E_{\mathrm{local}}, (i+1)\cdot E_{\mathrm{local}})$-th expert.

<details>
    <summary>Code: Stage construction snippet</summary>
    <pre><code class="language-python">

def get_expert_bounds(ep_rank: int, num_per_rank_experts: int):
    start_expert_id = ep_rank * num_per_rank_experts
    end_expert_id = start_expert_id + num_per_rank_experts
    return (start_expert_id, end_expert_id)

def get_model_shard_specs(dim: ParallelDims, cfg: Config):
    ...
    num_per_rank_experts = dim.num_experts // dim.ep_size
    start_expert_id, end_expert_id = get_expert_bounds(dim.ep_rank, num_per_rank_experts)
    ...

    spec = ModelShardSpec(
        ...
        per_rank_expert=num_per_rank_experts,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        ep_size=dim.ep_size,
        ep_group=dim.ep_group,
        ...
    )

    return spec  
    </code></pre>
</details>
<br>
On each rank, $E_{\mathrm{local}}$ experts are then initialized contiguously, for reasons we will get to later. 

<details>
    <summary>Code: Experts initialization</summary>
    <pre><code class="language-python">
class ExpertFFN(nn.Module):

    def __init__(self, cfg: ModelConfig, spec: ModelShardSpec):
        ...

        self.W_gate = nn.Parameter(
            torch.empty(spec.per_rank_expert, cfg.d_model, cfg.ffn_in, ...)
        )
        self.W_val = nn.Parameter(
            torch.empty(spec.per_rank_expert, cfg.d_model, cfg.ffn_in, ...)
        )
        self.W_out = nn.Parameter(
            torch.empty(spec.per_rank_expert, cfg.ffn_in, cfg.d_model, ...)
            )
        ...
    </code></pre>
</details>
<br>
The 0th training step starts with identical values for the replicated parameters and to ensure this, we broadcast the replicated parameters from EP rank = 0 to other ranks in the EP group <d-footnote>This is also the type of broadcasting we do in DP at the start of the training</d-footnote>.

The broadcast method is defined below and will prove to be useful once we begin composing these parallelisms together.

<details>
    <summary>Code: Broadcasting replicated parameters weight</summary>
    <pre><code class="language-python">
self.broadcast_parameters(
            params=groups["shared"],
            src_rank=self.dims.non_expert_dp_group_ranks[0],
            group=self.dims.non_expert_dp_group,
)
    </code></pre>
</details>


#### Producing assignments

The token representation on the source rank <d-footnote>local residual stream</d-footnote> with shape $\left[b, \, S, \,D\right]$ is flattened across batch and sequence length dimensions to $\left[T, \,D\right]$. The router scores every token against all $E$ experts, producing router scores with shape $[T,E]$, and top-$K$ selection gives the indices of the top-$K$ experts, `expert_IDs`, for each token as well as the probability with which each expert was chosen, `router_weights`. Both `router_weights` and `expert_IDs` have shape $\left[T, \,K\right]$.

Each token now has $K$ destinations and conceptually, the residual stream is expanded from $[T,D]$ to $[T,K,D]$ which is also flattened to $[TK, D]$. This $[TK,D]$ tensor is the **assignment buffer**. `router_weights` and `expert_IDs` are flattened to vectors of length $TK$.

In the assignment buffer, the $K$ assignments for token 0 come first, followed by the $K$ assignments for token 1, and so on. This *token-major* layout isn't convenient for All-to-All (A2A) communication as A2A requires the assignments be grouped by destination rank.

Let $c_e$ denote the number of assignments routed to expert $e$. We rearrange/pack the assignment buffer into *expert-major* order: the first $c_0$ assignments belong to expert 0, the next $c_1$ assignments belong to expert 1, and so on.<d-footnote>This is aligned with rank grouping because experts are mapped and initialized on ranks based on ID. So first $E_{local}$ experts are on rank 0 and the next $E_{local}$ experts are on rank 1, and so on.</d-footnote> The number of assignments is still $TK$, but the physical and logical interpretation of the tensors is changed. To implement this, we count the number of assignments routed to each expert, compute the offsets from their cumulative counts and write each assignment to the corresponding region of the packed buffer.


<!-- we allocate the memory location for all the tokens that wrote to each expert, map each token to some relative position for that expert and write to that memory location. -->

<details>
  <summary>Code: Producing assignments</summary>
  <pre><code class="language-python">
tokens_per_expert = torch.bincount(
    expert_idx.reshape(-1), minlength=cfg.num_experts
)
expert_offsets = torch.empty(cfg.num_experts + 1, dtype=torch.long, device=x.device)
expert_offsets[0] = 0
expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)

total_assignments = flat_tokens.shape[0] * self.cfg.top_k
packed_X = torch.empty(
    (total_assignments, d_model), dtype=flat_tokens.dtype, device=x.device
)
...
for expert in range(self.cfg.num_experts):
    indices = torch.argwhere(topk_expert_idx == expert)
    if indices.numel() == 0:
        continue
    start = expert_offsets[expert].item()
    end = expert_offsets[expert + 1].item()
    packed_X[start:end] = flat_tokens[indices[:, 0]]
    ...
  </code>
  </pre>
</details>


#### Dispatch, expert computation and combine

Each rank in the EP group has its assignment buffer grouped by destination expert and therefore by destination rank. Dispatching (or exchanging) the assignments, from All ranks to All ranks requires each rank knowing how many tokens it will send to and receive from every other rank.

<!-- Each rank in the EP group, with it's assignment buffer, exchanges tokens with every other rank<d-footnote>Ideally, every other rank. This motivates the need for balanced assignment as ranks that do not get any assignment do not utilize the GPU properly</d-footnote>, each expert on each rank runs the forward pass on the tokens aggregated from every rank in the EP group and returns the token representation to the respective ranks that sent them (this might be repetitive, maybe I said this elsewhere already, IDK). -->

<!-- Dispatching(or exchanging) the buffer, from All ranks to All ranks requires each rank knowing - aggregated from every other rank - how many tokens it would be receiving. -->

- For each expert $e$, we count how many tokens would be routed to it, `expert_counts`, shape is $[E]$.
- We reshape `expert_counts` to `send_matrix`, with shape $[P_{EP}, E_{local}]$, where the `ij`-th entry corresponds to the number of tokens the current rank wants to route to local expert `j` in rank `i`.
- We perform a small A2A exchange of these counts across the EP group. On each rank, the resulting `recv_matrix` also has shape $[P_{EP}, E_{local}]$, where the `ij`-th entry corresponds to the number of tokens rank i sent to local expert j on the current rank. <d-footnote>The row corresponding to the current rank represents locally routed assignments; that portion does not traverse the inter-GPU interconnect.</d-footnote>.
- From `recv_matrix`, we know both the total receive-buffer size, `sum(recv_matrix)` and how many assignments will arrive from each source rank, `recv_matrix.sum(dim=1)[i]`. We can then allocate `received_X` and perform A2A on the token payloads.
- After A2A, `received_X` is laid out primarily by source rank i.e assignments from rank 0 are contiguous, followed by assignments from rank 1, and so on. Expert computation instead requires assignments for each local expert to be contiguous. We therefore permute the received buffer from this source-major layout into an expert-major layout.
- The number of assignments for each local expert is given by `recv_matrix.sum(dim=0)`, and cumulative sums of these counts give the offsets into the expert-major buffer. Since both the expert weights and their assigned tokens are now contiguous, expert computation reduces to $E_{\mathrm{local}}$ independent MLP computations.

After the expert computation, we reverse the permutation to restore the source-major layout. A second A2A sends each assignment back to its source rank. We then use the saved token indices to undo the original packing, multiply each expert output by its router weight and sum the $K$ outputs for each token. This restores the residual stream to $[T,D]$.


#### EP backward pass

During the backward pass, expert gradients stay on their own expert owner rank and non-expert gradients are AllReduced across the EP group.

With a mean loss on each local data shard, these two gradient paths have different scale factors. An expert owner receives gradient contributions from every rank in its EP group, so those contributions add up, while gradients for replicated non-expert parameters are averaged by AllReduce. Expert gradients therefore need to be divided by $P_{EP}$ before gradient clipping and the optimizer step to keep both parameter groups on the same global-mean scale.

### Experimental Results

<figure id="fig-ep-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#ep"
    title="Expert-parallel experiment results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    EP results
  </figcaption>
</figure>


## DP + PP

Composing DP and PP relaxes the model-state memory while maintaining DP's throughput benefits. DP creates $P_{DP}$ model replicas, each getting a different data shard and PP shards each replica across $P_{PP}$ stages. The total number of ranks is therefore $P_{\mathrm{world}}=P_{DP} \cdot P_{PP}$.

This composition also maintains their individual communication patterns, albeit, slightly different. There is the activation transfer **within** each replica and gradient AllReduce **across** replica, for all the stages.

<figure id="fig-dp-pp-composition" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/dp-pp-composition.html' | relative_url }}"
    title="Composition of data and pipeline parallelism"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(360px, 67vw, 500px); border: 0;"
  ></iframe>
  <figcaption>
    2D DP x PP Composition
  </figcaption>
</figure>


### Implementation

#### Rank Identity

A rank must know which replica it belongs to and which stage within the replica it belongs to. This determines who it sends/receives activation from and who it all-reduces with. 

For rank $r\in\{0,\ldots,P_{\mathrm{world}}-1\}$, it maps to

\[
\begin{aligned}
r &= (\text{dp_rank}\cdot P_{PP})+\text{pp_rank} \\
\text{dp_rank} &= \left\lfloor\frac{r}{P_{PP}}\right\rfloor \\
\text{pp_rank} &= r \bmod P_{PP} \\
\end{aligned}
\]

where $\text{dp_rank}$ is the DP-replica coordinate and $\text{pp_rank}$ is the PP-stage coordinate.

From the PP section, each stage is aware of the rank of the next stage and previous stage.

#### Build the stage-local model


The model abstraction of treating a local model as just a collection of `nn.Modules` vs (Embed + PosEmbed + TransformerBlock + Unembed) proved useful here. The PP coordinate determines which modules a rank owns. All ranks with the same $p$ i.e $(0,p),(1,p),\ldots,(P_{DP}-1,p)$ construct the same stage parameters. The optimizer states are also with respect to the stage parameters.

<!-- ??? In intro, we don't talk about much fwd and bwd pass for individual parallelisms, would go indepth, for the compositions. -->
To ensure that training starts with identical parameters, each rank builds its own model parameters, but for each pipeline stage p, we broadcast from all the ranks with DP = 0 to the DP group containing all replicas of stage p.

#### Forward pass

GPUs with pp_rank = 0 across all replicas get disjoint input shards.
- Across replicas, there is no communication, each replica computes like normal DP.
- Within replicas, each stage computes with its local parameters and sends activations to the next stage

#### Backward pass

Each stage accumulates gradients across its microbatches and on the backward pass that completes the gradient accumulation across the microbatches for the step, rank $(d,p)$ AllReduces only with ranks holding the same stage $p$, so every replica of that stage receives the same averaged gradient.

If $g^{\mathrm{local}}_{d,p}$ is the gradient for stage $p$ in replica $d$, the reduced gradient is:

$$
\bar g_p
=
\frac{1}{P_{DP}}
\sum_{d=0}^{P_{DP}-1}g^{\mathrm{local}}_{d,p}.
$$

<details>
  <summary>Code: composed backward pass</summary>
  <pre><code class="language-python">for microbatch_id in reversed(range(num_microbatches)):
    output_grad = (
        None
        if pp_rank == pp_size - 1
        else recv_backward(microbatch_id, group=pp_group)
    )

    input_grad = backward_microbatch(
        microbatch_id,
        output_grad,
        # microbatch 0 is executed last in the reversed sweep.
        sync_gradients=microbatch_id == 0,
        dp_group=dp_group,
    )

    if pp_rank > 0:
        send_backward(
            microbatch_id,
            input_grad,
            group=pp_group,
        )

reducer.wait_for_allreduces()
clip_composed_gradient()
optimizer.step()</code></pre>
</details>

The optimizer boundary is now constrained by both axes. A stage cannot step
until it has completed backward for every microbatch **and** all of its DP
AllReduces have finished. Each rank then updates only its stage-local parameters;
matching stage replicas arrive at identical new values because they began with
the same values and use the same reduced gradients.

#### Gradient clipping

For gradient clipping, each rank first computes the squared gradient norm of its local stage. We sum these values across the PP group and take the square root, giving the norm of the complete pipeline model exactly once. We do not reduce this norm across the DP group because matching stages already have identical gradients after AllReduce. Every rank then uses this global norm to clip its local gradients before the optimizer step.

### Experimental Results

<figure id="fig-dp-pp-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#dp-pp"
    title="Data and pipeline parallel composition results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    DP + PP results
  </figcaption>
</figure>

## PP + EP
Here, we partition the model along two axes. PP assigns $\frac{L}{P_{PP}}$ to each stage while EP shards the expert within each stage across $P_{EP}$ ranks. A stage is now an EP group, rather than a single rank.

<figure id="fig-pp-ep-composition" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/pp-ep-composition.html' | relative_url }}"
    title="Composition of pipeline and expert parallelism"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(430px, 72vw, 740px); border: 0;"
  ></iframe>
  <figcaption>
    2D PP x EP Composition
  </figcaption>
</figure>

<!-- - We partition the model across it's layers and shard each stage's expert across more ranks. -->
<!-- - The notion of a stage here moves from a rank to multiple ranks i.e each stage is an expert group.  -->
<!-- - So assume a 2L model, each corresponding to different stages. Each layer then shards it's expert across N ranks -->
<!-- - Releaves memory pressure for the entire model and expert memory + activation pressure for EP. -->

### Implementation
#### Rank Identity
Each rank is identified by which PP stage it implements and which expert rank within the stage that it implements.

For rank $r\in\{0,\ldots,P_{\mathrm{world}}-1\}$, it maps to

\[
\begin{aligned}
r &= (\text{pp_rank} \cdot P_{EP}) + \text{ep_rank}, \\
\text{pp_rank} &= \left\lfloor \frac{r}{P_{EP}} \right\rfloor, \\
\text{ep_rank} &= r \bmod P_{EP}.
\end{aligned}
\]

These coordinates define two kinds of process groups. Ranks with the same `pp_rank` form an EP group and exchange tokens within a stage. Ranks with the same `ep_rank` form a PP group and pass residual-stream activations
between stages. For example, with $P_{PP}=P_{EP}=2$, ranks $(0,1)$ implement the first stage and ranks $(2,3)$ implement the second, while the PP connections are $0\leftrightarrow2$ and $1\leftrightarrow3$.

#### Map parameters to ranks
Model construction reuses the [stage-local model](#build-the-stage-local-model): `pp_rank` selects the range of transformer layers and `ep_rank` selects the experts owned within those layers. Non-expert parameters, including the router, are replicated across the stage's EP group, while expert parameters are sharded.

#### Forward pass + Backward pass
During the forward pass, each rank in the first stage receives a different input shard. Within a stage, routing, dispatch, expert computation, and combine proceed exactly as in EP. Once the assignments have been returned and combined, rank $(p,e)$ sends its local residual-stream shard to rank $(p+1,e)$. The final stage computes a loss for each local input shard.

The backward pass follows the same communication paths in reverse: activation gradients move between pipeline stages, while gradients for replicated non-expert parameters are averaged within each stage's EP group. Expert gradients remain local because every expert has a unique owner.

Expert gradients use the [same normalization as in EP](#ep-backward-pass).

### Experimental Results

<figure id="fig-pp-ep-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#pp-ep"
    title="Pipeline and expert parallel composition results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    PP + EP results
  </figcaption>
</figure>




<!-- Using the rank coordinate, we determine which layers are on each stage and which expert params are on which expert rank. -->
<!-- - Number of stages = $P_{PP}$, number of layers on each stage = $\frac{L}{P_{PP}}$ -->

<!-- - Each rank get it's identity, which is a coordinate of which pipeline stage and expert rank within that stage that it belongs to. -->
<!-- - Decide which layers belong to each rank and within each stage, which expert belongs to which rank, based on the rank's identity -->

<!-- - In the forward pass, the input to a stage is the same as in PP i.e tokens if stage is 0 and activations from stage i - 1 if stage > 0. -->
<!-- - Each stage is exactly an EP group and we already covered this in EP section -->
<!-- - All non-last stage send their activations to the next stage and the last stage compute the loss, from which we backpropagate. -->


## DP + EP

Here, DP creates $P_{DP}$ model replicas and each replica shards its experts across $P_{EP}$ ranks. This allows to increase throughput with DP while reducing expert parameter and activation memory pressure. The total number of ranks is $P_{\mathrm{world}}=P_{DP} \cdot P_{EP}$.

As with [DP + PP](#dp-pp), a replica is no longer a single rank. In this case though, it's an expert group that holds sharded expert parameters and replicated non-expert parameters.

<figure id="fig-dp-ep-composition" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/dp-ep-composition.html' | relative_url }}"
    title="Composition of data and expert parallelism"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(460px, 74vw, 760px); border: 0;"
  ></iframe>
  <figcaption>
    2D DP x EP Composition
  </figcaption>
</figure>

### Implementation

#### Rank Identity

A rank must know which replica it belongs to and which expert shard within the replica it implements. For rank $r\in\{0,\ldots,P_{\mathrm{world}}-1\}$, it maps to

$$
\begin{aligned}
r &= (\text{dp_rank} \cdot P_{EP}) + \text{ep_rank}, \\
\text{dp_rank} &= \left\lfloor \frac{r}{P_{EP}} \right\rfloor, \\
\text{ep_rank} &= r \bmod P_{EP}.
\end{aligned}
$$

Ranks with the same `dp_rank` form an EP group and exchange token assignments within one replica while ranks with the same `ep_rank` own copies of the same expert shard across different DP replicas and form an **expert DP group**. All the $P_{\mathrm{world}}$ ranks also form a **shared DP group**; this is because across a replica (EP group) and all replicas (DP group), non-expert parameters are replicated. So the size of the shared DP group is $P_{DP} \cdot P_{EP}$

#### Map parameters to ranks

The `ep_rank` determines which $\frac{E}{P_{EP}}$ experts a rank loads. Across the DP groups however, there are different replication patterns and this determines the broadcast pattern for the `state_dict()`. 
- Expert parameters are replicated across *expert DP group*, but sharded across EP group. This means at the start of the training, for each expert DP group, the rank with `dp_rank = 0` broadcasts its expert parameters to the other DP replicas with the same `ep_rank`.
- Non-expert parameters are replicated across every rank. Rank 0 broadcasts the non-expert parameters state_dict to every rank in the world_size.

#### Forward pass + Backward pass

Each rank gets a different input shard here. We achieve this by defining a `data_rank` and `data_world_size`. `data_world_size` is equivalent to $P_{EP} \cdot P_{DP}$ and `data_rank` is equivalent to $\text{dp_rank} \cdot P_{EP} + \text{ep_rank}$, and we pass these arguments to the `PackedTokenDataset`. Notice that this works for the DP case alone if we set $P_{EP}$ = 1.

There is no communication between DP replicas during the forward pass and within each replica, routing, dispatch, expert computation and combine proceed exactly as described in the EP section.

During the backward pass, every rank computes the backward pass from its own data shard. Expert parameters are AllReduced across their `expert_DP_group` and non-expert parameters are AllReduced across the `shared_DP_group` which turns out to be across all ranks. Expert gradients use the [same normalization as in EP](#ep-backward-pass).

We instantiate two reducers for each of the expert parameters and non-expert parameters and optimizer step occurs only after all the parameters have been AllReduced across their respective communication groups.

### Experimental Results

<figure id="fig-dp-ep-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#dp-ep"
    title="Data and expert parallel composition results"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    DP + EP results
  </figcaption>
</figure>

## DP + PP + EP

DP creates multiple model replicas, PP shards each replica across its layers and EP shards the experts within each pipeline stage. A complete DP replica is therefore a pipeline in which every stage is an EP group. The total number of ranks is $P_{\mathrm{world}} = P_{DP}P_{PP}P_{EP}$

This composition relaxes the three bottlenecks discussed so far: PP reduces the model state held by each stage, EP reduces the expert state held by each rank and DP increases the amount of data processed in parallel.

<figure id="fig-dp-pp-ep-composition" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/dp-pp-ep-composition.html' | relative_url }}"
    title="Composition of data, pipeline and expert parallelism"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(560px, calc(92vw + 80px), 1060px); border: 0;"
  ></iframe>
  <figcaption>
    3D DP x PP x EP Composition
  </figcaption>
</figure>

### Implementation

#### Rank Identity

Each rank is identified by a coordinate
$(\text{dp_rank},\text{pp_rank},\text{ep_rank})$. We keep `ep_rank` as the
fastest-changing coordinate, followed by `pp_rank`:

$$
\begin{aligned}
r &= ((\text{dp_rank}\cdot P_{PP}) + \text{pp_rank})P_{EP}
     + \text{ep_rank}, \\
\text{dp_rank} &= \left\lfloor
\frac{r}{P_{PP}P_{EP}}
\right\rfloor, \\
\text{pp_rank} &= \left\lfloor\frac{r}{P_{EP}}\right\rfloor
\bmod P_{PP}, \\
\text{ep_rank} &= r \bmod P_{EP}.
\end{aligned}
$$

The three coordinates determine all the communication groups:

- An **EP group** fixes `dp_rank` and `pp_rank`, and varies `ep_rank`. It contains the ranks implementing one pipeline stage in one DP replica.
- A **PP group** fixes `dp_rank` and `ep_rank`, and varies `pp_rank`. It forms one path through the stages of a pipeline replica.
- An **expert DP group** fixes `pp_rank` and `ep_rank`, and varies `dp_rank`. It contains replicas of the same expert shard at the same pipeline stage.
- A **non-expert DP group** fixes `pp_rank`, and varies both `dp_rank` and `ep_rank`. It contains every copy of the non-expert parameters belonging to that stage.

#### Map parameters to ranks

We again reuse the stage-local model construction. The `pp_rank` selects the range of transformer layers, the `ep_rank` selects the experts within those layers and the `dp_rank` creates another copy of that stage and expert shard.

Consequently, an expert parameter is sharded across both PP and EP and is only replicated across DP. A non-expert parameter is sharded across PP, but is replicated across both DP and EP. The optimizer on each rank is constructed only from the parameters that rank owns.

#### Forward pass + Backward pass

At the start of the forward pass, ranks with `pp_rank = 0` receive disjoint input shards. Each DP replica runs its own GPipe schedule. For a given microbatch, the ranks in a stage first perform the EP routing, dispatch, expert computation and combine. Rank $(d,p,e)$ then sends its local residual-stream activation to rank $(d,p+1,e)$. The last stage computes a loss for each local input shard.

The coordinates make the separation between the communication patterns
explicit:

- A2A communication happens within an EP group and therefore does not cross a pipeline stage or DP replica.
- Pipeline activations and activation gradients move within a PP group and therefore keep their `dp_rank` and `ep_rank` coordinates.
- Gradient AllReduce happens across parameter replicas, using the expert or non-expert DP group depending on the parameter.

The backward pass follows the forward communication paths in reverse. After all the microbatches have completed backward, expert gradients are averaged across the corresponding expert DP groups and non-expert gradients are averaged across the non-expert DP groups. Each rank can then update its local stage parameters.

Expert gradients use the [same normalization as in EP](#ep-backward-pass).

Ranks in the same EP stage must also execute the same microbatch in the same order. This is because every rank in the EP group has to enter the count and payload A2A collectives in the same order, even though the number of assignments sent to each expert may be different.

No new communication primitive is introduced by the 3D composition. It is the same point-to-point PP transfers, EP All-to-Alls and DP AllReduces from the individual implementations. The additional complexity is in constructing the correct groups and ensuring that each parameter is synchronized only with ranks that own a copy of it.

### Experimental Results

<figure id="fig-three-axis-results" class="l-page">
  <iframe
    src="{{ '/assets/_draft/distributed/distributed-results.html' | relative_url }}#three-d"
    title="Three-axis parallelism result in context of the other 8-GPU runs"
    loading="lazy"
    scrolling="no"
    style="display: block; width: 100%; height: clamp(730px, calc(1000px - 35vw), 880px); border: 0;"
  ></iframe>
  <figcaption>
    DP + PP + EP results
  </figcaption>
</figure>


## Discussion, Future work

The model used for benchmarking fits on a single GPU, hence, the reason DP beats other composition for the same number of GPUs. The memory used during training varies significantly, especially with PP. With 4x PP, we could easily 4x the batch size per step and 2x the model and we still don't get OOM. However, the tradeoff here is that this will likely take more time.

We rented GPUs from vast.ai and we can't always ensure the same baseline perf characteristics across all sessions. One main reason is that different providers allow different topologies between the GPUs and this was significant especially when running the 4-GPU and 8-GPU experiments as we ran into multiple crashes and failed runs. Our approach to this was to stick to a specific GPU e.g RTX 3090 from a specific provider ID.

Performance is very intertwined with distributed training and at the time of writing, with commit `b3db5c6`, there is still a significant room for performance improvement. For one, the GEMM kernels are very basic and we do not take advantage of the tensor cores in the kernels. Other opportunities that are actively being explored are attention kernels, capacity load balancing for the experts, fusing the layernorm with other ops, etc. We would explore end-to-end performance in a separate blogpost.

Other future work to be explored also include
- Automatically choosing some parallelism composition based on a static model specs and hardware specs
- During training, adjusting the parallelism composition on the fly, based on online scaling of GPUs


## Acknowledgements

I'm very grateful to David Daniel for funding the compute for the experiments.

I also found so many materials useful when I was starting with the project,
- [Simon Boehm's](https://siboehm.com/) blogs on DP, PP optimizing matmul.
- [Lilian Weng's](https://lilianweng.github.io/posts/2021-09-25-train-large/) blog on training parallelism.
- [Jax scaling book](https://jax-ml.github.io/scaling-book/) on building nice intuition for the communication patterns and theoretical cost accounting.
- [ChatGPT](https://chatgpt.com/) for its infinite patience in answering clarifying questions, working through concepts and helping with the initial draft of the blogpost.
