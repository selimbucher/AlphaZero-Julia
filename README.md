# AlphaZero-Julia

An AlphaZero-style engine (self-play deep RL + Monte Carlo Tree Search) built **from
scratch in Julia**, playing Connect Four on a single home PC with one NVIDIA GPU.
Written as my matura thesis (Oct 2023, before AI coding assistants were useful) —
[thesis PDF, German](https://slimmer.ch/AlphaZero-Julia.pdf).

## The interesting part: fighting the CPU→GPU bottleneck

Profiling showed most wall-clock time wasn't spent computing — it was spent shuttling
data between CPU and GPU for the network evaluations MCTS requests one position at a
time. Two mechanisms fix this:

- **DecayCluster batching** (own design, implemented here): tree-search leaves are
  collected and evaluated as whole clusters instead of individually. Some cluster members
  turn out to be wasted work, but far fewer transfers make it a net win; cluster sizes
  decay as searches get deeper, controlled by a decay function.
- **Multithreaded, GPU-parallel MCTS** (designed in the thesis, not implemented in this
  repo): multiple workers expanding the tree concurrently without blocking on GPU
  results, coordinated via Julia `Channel`s and atomics.

Result: an engine that trains and plays on consumer hardware — it learns winning
strategies and blocks opponent threats, though it does not play perfectly.

## Running it

Requires Julia and an NVIDIA GPU (CUDA). Install the dependencies (Flux, CUDA, cuDNN,
BSON, FileIO, ProgressMeter, Plots):

```
julia src/dependencies.jl
```

Then, from `src/`:

| command | what it does |
|---|---|
| `julia train.jl` | self-play training (parameters persist in `model.bson`; delete to reset) |
| `julia humanvsai.jl` | play against the engine |
| `julia performance.jl` | reproduce the batching experiment from thesis §5.1 |

AMD GPUs may work via `AMDGPU.jl` (untested).
