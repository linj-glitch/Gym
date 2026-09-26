# Claude Code sandboxed agent

The sandboxed sibling of `responses_api_agents/claude_code_agent`. It speaks the
resources-server-owned-sandbox contract used by the `swebench` and `deepswe`
resources servers — `/seed_session` creates the task sandbox, this agent
connects to it, runs the Claude Code CLI inside, and `/verify` grades the
working-tree diff — the same flow as `opencode_sandboxed_agent`, with the CLI
conventions and stream-json parsing of `claude_code_agent`.

## How model traffic flows

Claude Code speaks the Anthropic Messages API, so its requests go directly from
the sandbox to `anthropic_base_url` (`/v1/messages` is appended by the CLI); no
Gym model server is on the path. Point it at a
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) proxy: Switchyard
translates Messages-API traffic to whatever the routed target speaks (for
example a vLLM chat-completions endpoint, mapping the model's separate
reasoning field to and from `thinking` blocks) and can route across targets.
Each rollout's Gym session id is sent as `ANTHROPIC_CUSTOM_HEADERS:
x-switchyard-session-id`, so routing decisions join back to rollouts and
session-affine algorithms (escalation latching) behave per task.

To host that proxy from the same Gym run, start
`responses_api_models/switchyard_model` with a fixed `proxy_port` and set
`anthropic_base_url` to `http://127.0.0.1:<proxy_port>`; the enroot and
Docker sandbox providers share the host network namespace, so the loopback
address resolves inside the sandbox.

## Installing the CLI in the sandbox

By default the network installer runs once per sandbox, pinned by
`claude_code_version`. For repeatable builds and nodes without egress, mount or
upload a binary and set the pair
`remote_claude_install_script_path` / `remote_claude_binary_path` (both or
neither); the script is run as `bash <script> --binary <binary>` and is
expected to put `claude` on `PATH`, e.g. by symlinking into `/usr/local/bin`.

## Configuration

| Key | Meaning |
| --- | --- |
| `anthropic_base_url` | Endpoint root the CLI talks to (env `CLAUDE_CODE_ANTHROPIC_BASE_URL`). |
| `anthropic_auth_token` | Bearer the CLI sends; a placeholder for a local proxy. Never a real provider key. |
| `claude_model` | Model the CLI requests — a Switchyard route id. Defaults to `${policy_model_name}`. |
| `claude_max_turns`, `claude_max_output_tokens` | Optional caps (`--max-turns`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`). |
| `bare` | `--bare`: skip hooks, plugins, MCP, memory and `CLAUDE.md` discovery. |
| `session_header_name` | Header carrying the rollout session id; `null` disables. |
| `sandbox_provider`, `sandbox_config`, `sandbox_timeout` | Same contract as `opencode_sandboxed_agent`. |

The run verdict is carried on the verify response as `claude_status`
(`completed`, `error`, `timeout`, ...) with `claude_num_turns`,
`claude_is_error` and the raw CLI stdout/stderr.

## Example

```bash
CLAUDE_CODE_ANTHROPIC_BASE_URL=http://127.0.0.1:8100 \
gym eval prepare \
  --config benchmarks/swebench/verified/claude_code.yaml \
  --config responses_api_models/switchyard_model/configs/switchyard_model.yaml \
  --config nemo_gym/sandbox/providers/enroot/configs/enroot.yaml \
  ++policy_model_name=switchyard/nemotron-3-nano \
  ++policy_model.responses_api_models.switchyard_model.proxy_port=8100
```
