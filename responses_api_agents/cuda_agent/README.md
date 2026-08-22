# CUDA Agent

Agentic CUDA/Triton kernel optimization. A thin specialization of
[`opencode_agent`](../opencode_agent): it seeds a per-rollout problem dir from
the task row, runs one agent session over it, and submits the final kernel to
the [`cudagym`](../../resources_servers/cudagym) resources server for the
canonical correctness-gated reward.

## What `run()` does

1. **Seed** the per-rollout `problem/` from the row's `verifier_metadata`
   (`definition.json` + `workload.jsonl`). The OpenCode config travels as the
   `OPENCODE_CONFIG_CONTENT` env var — never a workspace `opencode.json` — so
   the policy apiKey is not model-readable. The sandbox's `cudagym` CLI is
   pointed at the evaluation endpoint for the row's own `target_hardware`
   (`cudagym_endpoints`, else `CUDAGYM_UNIFIED_SERVER_URL`) and inherits the
   evaluation credentials the SDK reads from the environment (the `CUDAGYM_*`
   and `MODAL_PROXY_TOKEN_*` variable families), so the model gets its own
   `cudagym evaluate` feedback from the GPU its kernel is scored on.
2. **Run** one agent session (see the two profiles below); parse OpenCode's
   sqlite session into a Gym trajectory (`opencode_agent.parse_opencode_session`).
3. **Submit**: read the final kernel file (`kernel.py` / `main.cpp` by language)
   and append it as a closing assistant message (fenced) so the resources
   server scores the *submitted* kernel — the agent can't game the reward by
   self-selecting easy workloads in its own feedback calls.
4. **Verify**: POST the trajectory to the `cudagym` server's `/verify`.

## Sandbox profiles (`sandbox_profile`)

- **`minimal`** (default): bare `problem/` sandbox + a short contract prompt;
  we launch `opencode run` ourselves. `mount_namespace: true` gives each
  rollout production's literal paths (cwd `/workspace`, problem
  `/tmp/problem`); `hide_job_mounts: true` hides the job's dataset, model, and
  code bind mounts by mounting empty directories over them inside the
  namespace. The subprocess environment is built from
  `_SANDBOX_ENV_ALLOWLIST` rather than inherited, so the training job's
  unrelated credentials (Weights & Biases, Hugging Face, cluster tokens) never
  reach the model.
- **`solswarm`** — container mode: each rollout runs inside a real instance of
  the published SolSwarm agent image, extracted once per node and entered
  through the host's enroot runtime (`solswarm_container.py`). The image's own
  `entrypoint.sh` installs the skills, stitches `submission_rule.md`, renders
  the optimizer role prompt, writes the OpenCode config and launches OpenCode
  with its own flags; we contribute only the environment variables the pod
  controller would inject (`build_env`, delivered as a sourced file because
  `enroot start` resets the calling environment). The policy model is routed
  through the entrypoint's own model-registry machinery
  (`solswarm_container.write_registry_catalog_files`): the harness writes two
  JSON files into each rollout's tree — a model catalog declaring one
  synthetic model (`rl-policy`, `route.policy: registered_endpoint`) and an
  endpoint registry whose entry carries `agent_base_url` = the per-rollout
  capture URL `<model_server>/ng-rollout/<id>[/training-token-capture]/v1` —
  and the env contract
  points the entrypoint at them (`SOLSWARM_AGENT_MODEL_CATALOG_FILE` /
  `SOLSWARM_AGENT_MODEL_ENDPOINTS_FILE`, with `SOLSWARM_AGENT_MODEL` naming
  the synthetic model and `SOLSWARM_AGENT_MODEL_API_KEY` carrying the policy
  apiKey). The entrypoint reads those files in place of its baked-in registry
  and configures OpenCode to dial `agent_base_url` directly, so the
  rollout path prefix — and with it per-call token capture — survives the
  hop. Requires `enroot` and `gawk`
  on PATH, an agent-image squashfs (`enroot_image` or env
  `CUDA_AGENT_ENROOT_IMAGE`), and a `model_server` (all validated at server
  startup — there is no Python fallback). Also needs a solswarm checkout for
  the stitched problem extras: `solswarm_surface.root` or env
  `SOLSWARM_SURFACE_ROOT`.

Independent of profile, `submission_mode: solswarm_submit` installs SolSwarm's
real `/submit` skill and points `PLATFORM_API_ENDPOINT` at the cudagym
resources server's rollout-scoped `/rollout/<token>/api/v1` endpoints (202 +
operations polling, the same contract submit.py speaks in production).
Submissions are canonically evaluated and recorded server-side; `verify()`
rewards the best (or final) one; the final-kernel-file fallback is configurable
server-side.

[`cudagym_cuda_agent_solswarm.yaml`](../../resources_servers/cudagym/configs/cudagym_cuda_agent_solswarm.yaml)
turns both on. The agent-image squashfs is built from the pinned solswarm
checkout's `docker/Dockerfile.agent` for the cluster's CPU architecture.

## Sandbox check (once per job)

After its agent exits, the first rollout of a job records every file path
visible inside its sandbox. In container mode that listing is compared for
set equality against a reference listing from a container run of the same
extracted image (`solswarm_container.reference_tree`, cached per solswarm
commit and per image). The
verdict is printed once ("sandbox tree vs SolSwarm reference:
IDENTICAL|DIFFERS") and both listings land in
`$CUDA_AGENT_MANIFEST_DIR/sandbox_tree.json`. Files the model wrote under
model-writable roots count as work product, not sandbox drift. The listing
script and its output live in a directory mounted read-write into the agent's
own container, so the verdict is an integrity alarm against accident and
drift, not a defense against an agent that deliberately tampers with the
listing.

## Training (token capture)

Compose the base config with
[`cudagym_cuda_agent_training.yaml`](../../resources_servers/cudagym/configs/cudagym_cuda_agent_training.yaml)
and `vllm_model_for_training.yaml`: the run's rollout id (derived from the
row's task/rollout indices) prefixes the model URL as
`<model_server>/ng-rollout/<id>/training-token-capture/v1` (in the minimal
profile via the OpenCode provider baseURL, in container mode via the registry
files described above). The model server's capture middleware
(`nemo_gym/token_id_capture`) durably records each call's token ids; rollout
collection rebuilds the on-policy trajectory into `response.output` at
finalization and masks the sample (`mask_sample: true`) when capture is
incomplete or ambiguous. Two constraints: OpenCode compaction must stay OFF
(a mid-run compaction silently drops post-compaction turns from the token-id
trajectory; the shipped config disables it), and generation temperature/top_p
must be 1.0 (the training vLLM server asserts this). Without the overlay
(`model_server` unset) the minimal profile is eval/rollout-collection only (no
token ids); the solswarm profile refuses to start.

## Quick start (eval, minimal profile)

```bash
ng_run "+config_paths=[resources_servers/cudagym/configs/cudagym_cuda_agent.yaml,\
        responses_api_models/openai_model/configs/openai_model.yaml]"
ng_collect_rollouts +agent_name=cudagym_cuda_agent \
  +input_jsonl_fpath=resources_servers/cudagym/data/example.jsonl \
  +output_jsonl_fpath=cuda_rollout.jsonl +limit=2
```

Config fields beyond those documented for
[`opencode_agent`](../opencode_agent#config-fields): `resources_server` (the
`cudagym` server), `model_server` (training only; enables token capture),
`sandbox_instructions` (extra text appended to the minimal-profile contract),
`sandbox_profile`, `enroot_image`, `submission_mode`, `mount_namespace`,
`hide_job_mounts`,
`solswarm_reasoning_variant`, the `solswarm_surface` block (`root`,
`ban_framework_kernels`, `ban_cuda_graphs`, `banned_libraries`), and two
test/debug knobs: `entrypoint_override` (bind-mount a host `entrypoint.sh`
over the image's own, so a modified entrypoint can be exercised without
rebuilding the agent image) and `keep_transcripts` (persist every rollout's
entrypoint transcript to the manifest dir, not only failed ones). Per-row
`verifier_metadata` carries the KernelFactory problem — see the `cudagym`
resources server README.
