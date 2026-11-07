# Terminal-Bench 4.0 (local task tree, Harbor's published images)

[Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0): 66 terminal tasks across software, ML,
science, operations, security, hardware and media, graded by each task's held-out `tests/test.sh`. This server
runs the official task packages on a **single-container sandbox provider** (enroot, docker) without a Harbor Hub
account, an OpenSandbox deployment or any image build: Harbor publishes one agent image and one verifier image
per task on Docker Hub (`harborframework/terminal-bench:<task>-environment-v4.0.0` /
`<task>-verifier-v4.0.0`), and the task tree (`task.toml`, `instruction.md`, `tests/`, `solution/`) is the
`v4.0.0` tag of the mirror https://huggingface.co/datasets/harborframework/terminal-bench.

The pinned `resources_servers/terminal_bench_4` server is the OpenSandbox implementation (Compose, GPU tasks,
split CPU/GPU endpoints). This one covers the **52 CPU tasks**; `prepare.py` leaves Compose, GPU and
MCP-declaring tasks out and `/seed_session` refuses them.

## Contract

Harness-owned sandbox, as DeepSWE: `/seed_session` starts the agent container from the task's environment
image (image `WORKDIR`, `/logs/{agent,verifier,artifacts}` created) and returns its descriptor; the agent server
connects, runs its CLI inside and posts `/verify`. `/verify` runs the task's `[[verifier.collect]]` hooks, tars
every declared `artifacts` entry (honouring `exclude`) out of the agent container, stops it, starts the verifier
container (`environment_mode = "separate"`, every 4.0 CPU task) or reuses the agent container (`"shared"`),
restores the artifacts at their original paths, uploads `tests/` to `/tests` and runs `bash /tests/test.sh`
under the task's official `verifier.timeout_sec` (times `verifier_timeout_multiplier`). `reward` is the official
value from `/logs/verifier/reward.json` (else `reward.txt`); `passed` is `reward >= 1`. A grader that produced no
reward sets `evaluation_completed = false` and `failure_reason`.

Golden check without a model: `is_verifying_golden_solution: true` makes `/verify` start a fresh agent
container, run `solution/solve.sh`, then grade as above.

## Deviations from the official protocol

- Agent budget: tasks declare 8 h (`agent.timeout_sec = 28800`, and the instruction says so); the harness's
  `sandbox_timeout` is the effective cap. Report the cap used.
- Sandbox resources: the task's `cpus`/`memory_mb`/`storage_mb` are requested from the provider (times the
  `task_*_multiplier` knobs); whether they are enforced depends on the provider.
- Images are selected by tag; record the Docker Hub digests of the imported images beside the results.

## Run

```bash
export TB4_TASKS_DIR=/data/terminal-bench-v4.0.0/tasks   # or ++prepare_script_args.tasks_dir=...
gym eval prepare --benchmark terminal_bench_4_local/claude_code
gym eval run --benchmark terminal_bench_4_local/claude_code \
  --config nemo_gym/sandbox/providers/enroot/configs/enroot.yaml \
  --output results/tb4/rollouts.jsonl --concurrency 8
```

Pre-import the 104 images into the enroot squashfs cache (`sandbox.enroot.create.sqsh_cache_dir`, key
`sha256(image)[:16].sqsh`) so first-touch tasks do not pay the import.

## Licensing

Code: Apache-2.0. Terminal-Bench 4.0 tasks and images: Apache-2.0 (see the dataset's `LICENSE`).
