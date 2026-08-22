# CudaGym

This server is the verifier for the agentic kernel-optimization environment.
The tasks are CUDA/Triton/Python kernel problems from
[CudaGym](https://gitlab-master.nvidia.com/atlas/solswarm/-/tree/main/cudagym) in KernelFactory form
(a `definition.json` plus a `workload.jsonl`). This server pairs with
the [`cuda_agent`](../../responses_api_agents/cuda_agent) harness, which writes
and iterates on a kernel inside a per-rollout sandbox and submits the result.

## Reward

`verify()` runs a canonical CudaGym evaluation of the submitted kernel — every
workload in the row, fixed `EvalConfig` — on the configured eval server. The
agent's own in-loop `cudagym evaluate` runs are feedback only; the reward comes
solely from this server-side evaluation, so the agent cannot raise its score by
evaluating only easy workloads.

The reward is correctness-gated:

* A kernel that is not numerically correct on **every** workload earns exactly
  0. This covers compile errors, runtime errors, wrong numerics, and
  evaluations flagged `REWARD_HACK`. The verify response still reports
  `compiled` and `executed` flags for metrics, but no reward attaches to them
  (a bare stub "compiles" in JIT languages such as Triton).
* A correct kernel earns `reward_weights.correctness +
  reward_weights.performance * perf_term`, where the performance term is in
  [0, 1]. With the default weights the reward range is [0, 2].

The performance term is the mean per-workload **speed-of-light (SOL) score**,
the metric SolSwarm scores on. Per workload,
`S = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))`, clamped to [0, 1], where `T_k`
is the kernel's measured latency, `T_b` the human-best anchor latency, and
`T_SOL` the speed-of-light (hardware-limit) anchor latency. `S` is 0.5 at
human-best and 1.0 at speed-of-light. The anchors arrive per workload in
`verifier_metadata.sol_anchors` and are, by contract, measurements taken on the
row's own `target_hardware` — an anchor records no GPU, so this server cannot
check it; the dataset builder must attach only anchors measured on the row's
GPU.

For rows without anchors, a correct kernel can instead earn the performance
term from a log-normalized speedup over the timed reference. This fallback
requires both `perf_reward_config.allow_speedup_fallback: true` and an
evaluation that timed the reference (`benchmark_config: {benchmark_reference:
true}`, which costs an extra reference run per evaluation). Without those, an
anchor-less row earns only the correctness weight.

The reward computation is kept in sync with the single-turn baseline in
`nemo_rl/environments/atlas/reward.py` and
`nemo_rl/environments/atlas/cudagym_client.py` (NeMo-RL repository), so the
agentic and single-turn rewards are comparable.

## Task row format

One JSON object per line. `verifier_metadata` carries a KernelFactory problem —
a problem's `definition.json` object plus the rows of its `workload.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "<task prompt>"}]},
 "verifier_metadata": {
   "language": "triton",                 // a cudagym SupportedLanguages value
   "target_hardware": "B200",            // a cudagym SupportedHardware value
   "destination_passing_style": false,   // false: run() returns outputs; true: run() writes into passed buffers
   "definition": { /* the problem's definition.json object */ },
   "workloads": [ /* rows of its workload.jsonl */ ],
   "sol_anchors": { /* optional: {workload_uuid: {human_best_latency_ms, sol_latency_ms}} */ }
 }}
```

See `data/example.jsonl` for a runnable `vector_add` example; any script may
emit rows in this format (`agent_ref` included makes the JSONL directly
usable without `ng_prepare_data`).

## Hosting

`endpoints` maps each GPU the server scores on, named by its CudaGym
`SupportedHardware` value, to that GPU's evaluation endpoint. A task row is
evaluated on the endpoint for its own `target_hardware`, and a row naming a GPU
with no entry is refused. Each URL is passed to the SDK `Client` as both the
compile and GPU URL; an empty URL is resolved at runtime from the
`CUDAGYM_UNIFIED_SERVER_URL` or `CUDAGYM_URL` environment variable, which is how
in-allocation hosting supplies a load-balancer address that does not exist yet
when the job is submitted. The SDK resolves credentials per
target from the environment: Modal hosts read `MODAL_PROXY_TOKEN_ID` and
`MODAL_PROXY_TOKEN_SECRET`; the SolSwarm platform proxy reads `API_TOKEN`.
For agentic RL, the eval GPUs
must be dedicated (not time-shared with generation) so latency measurements are
not contended.

With `verify_endpoint_sku: true` (the default), the server checks each
endpoint's `/health` once, at that endpoint's first use: if the reported
`gpu_model` or `sm_version` does not match the key it is filed under,
evaluation requests fail loudly instead of silently timing kernels on the wrong
GPU.

## Submission endpoints (solswarm_submit mode)

When `cuda_agent` runs with `submission_mode: solswarm_submit`, this server
also exposes rollout-scoped endpoints that implement SolSwarm's platform API,
so SolSwarm's unmodified `submit` skill works against it:

* `POST /rollout/<token>/begin` — the agent registers the task's
  `verifier_metadata` before launching the sandbox.
* `POST /rollout/<token>/api/v1/evaluate` — accepts a solution bundle and
  returns HTTP 202 with an operation id. The canonical evaluation runs in the
  background and its result is recorded server-side; nothing produced inside
  the sandbox is trusted for reward.
* `GET /rollout/<token>/api/v1/operations/<id>` — polled by submit.py until
  the evaluation finishes.
* `POST /rollout/<token>/api/v1/agents/trace`, `/insights`, `/bug-reports` —
  acknowledged and dropped, so SolSwarm's end-of-run uploads and helper skills
  do not error or retry.

`verify()` then rewards the best (`reward_mode: best_submission`, SolSwarm's
behavior) or the last (`final_submission`) recorded submission and reports
`n_submissions`, `best_submission_reward`, and `final_submission_reward`.
Scoring the final kernel file from the trajectory remains the fallback when
nothing was submitted, unless `final_file_fallback: false` (production scores
only submissions).

## Configuration

| Field | Meaning |
|---|---|
| `endpoints` | GPU SKU to evaluation endpoint; a row is scored on the endpoint for its `target_hardware` (see Hosting). |
| `verify_endpoint_sku` | Check each endpoint's `/health` GPU against its `endpoints` key once, at first use. |
| `compilation_timeout` | Seconds allowed for compilation per evaluation. |
| `execution_timeout_per_trial` | Seconds allowed per workload; multiplied by the workload count. |
| `reward_weights` | `correctness` and `performance` weights. |
| `perf_reward_config` | `clip_max`, `clip_min`, `speedup_ratio` for the speedup fallback, and `allow_speedup_fallback`. |
| `benchmark_config` | CudaGym `EvalConfig` overrides (e.g. `benchmark_reference`). |
| `reward_mode` | solswarm_submit mode: `best_submission` or `final_submission`. |
| `max_submissions` | solswarm_submit mode: canonical evaluations one rollout may spend. |
| `submission_ttl_seconds` | solswarm_submit mode: how long unclaimed rollout state is kept. |
| `final_file_fallback` | solswarm_submit mode: whether an unsubmitted rollout's final kernel file may be scored. |

See `configs/cudagym_cuda_agent.yaml` (base), `configs/cudagym_cuda_agent_training.yaml`
(token-id capture for training), and `configs/cudagym_cuda_agent_solswarm.yaml`
(SolSwarm container mode plus the submission flow).
