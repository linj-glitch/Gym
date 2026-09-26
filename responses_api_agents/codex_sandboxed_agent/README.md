# Codex sandboxed agent

The sandboxed sibling of `responses_api_agents/codex_agent`. It speaks the
resources-server-owned-sandbox contract used by the `swebench` and `deepswe`
resources servers — `/seed_session` creates the task sandbox, this agent
connects to it, runs `codex exec --json` inside, and `/verify` grades the
working-tree diff — the same flow as `opencode_sandboxed_agent` and
`claude_code_sandboxed_agent`, with the generated `config.toml` and JSONL
parsing of `codex_agent`.

## How model traffic flows

Codex speaks the OpenAI Responses API over SSE, so its requests go directly
from the sandbox to `openai_base_url` (which must include `/v1`; the CLI
appends `/responses`); no Gym model server is on the path. Point it at a
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) proxy: Switchyard
accepts Responses traffic, including Codex's custom tools and namespaced tool
names, translates it to whatever the routed target speaks (for example a vLLM
chat-completions endpoint) and streams the reply back as Responses events. Each
rollout's Gym session id rides on the provider's `http_headers` as
`x-switchyard-session-id`, so routing decisions join back to rollouts and
session-affine algorithms (escalation latching) behave per task.

To host that proxy from the same Gym run, start
`responses_api_models/switchyard_model` with a fixed `proxy_port` and set
`openai_base_url` to `http://127.0.0.1:<proxy_port>/v1`; the enroot and Docker
sandbox providers share the host network namespace, so the loopback address
resolves inside the sandbox.

## Installing the CLI in the sandbox

By default the pinned GitHub release tarball (`codex_version`, static musl
build) is downloaded once per sandbox. For repeatable builds and nodes without
egress, mount or upload a binary and set the pair
`remote_codex_install_script_path` / `remote_codex_binary_path` (both or
neither); the script is run as `bash <script> --binary <binary>` and is expected
to put `codex` on `PATH`, e.g. by symlinking into `/usr/local/bin`.

## Configuration

| Key | Meaning |
| --- | --- |
| `openai_base_url` | Endpoint the CLI talks to, with `/v1` (env `CODEX_OPENAI_BASE_URL`). |
| `openai_api_key` | Bearer the CLI sends; a placeholder for a local proxy. Never a real provider key. |
| `codex_model` | Model the CLI requests — a Switchyard route id. Defaults to `${policy_model_name}`. |
| `model_context_window`, `model_max_output_tokens` | Window and reply cap for a model Codex has no catalog entry for. |
| `reasoning_effort` | `model_reasoning_effort`; `null` omits it. |
| `stream_idle_timeout_ms` | Provider idle budget; defaults to `sandbox_timeout` so a non-streaming upstream behind the proxy fits. |
| `session_header_name` | Header carrying the rollout session id; `null` disables. |
| `extra_config` | Deep-merged over the generated `config.toml`. |
| `sandbox_provider`, `sandbox_config`, `sandbox_timeout` | Same contract as `opencode_sandboxed_agent`. |

Codex runs with `approval_policy = "never"` and
`sandbox_mode = "danger-full-access"`: the task container is the sandbox, and
Codex's own landlock/bwrap sandboxing is not available inside it.

The run verdict is carried on the verify response as `codex_status`
(`completed`, `error`, `no_output`, `exec_error`) with `codex_errors` and the
raw CLI stdout/stderr.

## Example

```bash
CODEX_OPENAI_BASE_URL=http://127.0.0.1:8100/v1 \
gym eval prepare \
  --config benchmarks/swebench/verified/codex.yaml \
  --config responses_api_models/switchyard_model/configs/switchyard_model.yaml \
  --config nemo_gym/sandbox/providers/enroot/configs/enroot.yaml \
  ++policy_model_name=switchyard/nemotron-3-nano \
  ++policy_model.responses_api_models.switchyard_model.proxy_port=8100
```
