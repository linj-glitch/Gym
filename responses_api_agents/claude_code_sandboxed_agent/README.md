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
| `bare` | `--bare`: skip hooks, plugins, MCP, memory and `CLAUDE.md` discovery. It also leaves the CLI three tools (Bash, Edit, Read). With `bare: false` the CLI gets its full tool set (Agent, Workflow, Write, Grep, Glob, ...) under `--setting-sources user --strict-mcp-config`, so a repository's own `.claude/settings.json` or `.mcp.json` cannot add hooks or servers. |
| `disallowed_tools` | Withheld from the full tool set (`--disallowed-tools`); default `[WebSearch, WebFetch]`. WebSearch is Anthropic's server-side tool and returns nothing through a proxy; WebFetch would let a benchmark rollout read the upstream fix online. |
| `ultracode` | Claude Code's ultracode mode: `--effort ultracode` (xhigh effort plus a standing instruction to use the Workflow tool, which runs subagents in parallel) with `CLAUDE_CODE_WORKFLOWS=true`. Needs `bare: false`. Subagents call the same endpoint; Claude Code marks their requests with an `x-claude-code-agent-id` header, which a Switchyard route's `subagents` policy can send to another target. |
| `workflow_max_concurrent_agents` | `CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS`; `null` keeps the CLI default. Each concurrent agent is one more stream against the endpoint, so size `num_samples_in_parallel` with it. |
| `claude_max_context_tokens` | `CLAUDE_CODE_MAX_CONTEXT_TOKENS`: the served context window, so auto-compaction fires relative to it for a model id the CLI does not know. |
| `claude_max_thinking_tokens` | `MAX_THINKING_TOKENS`: cap on thinking per reply. The reply cap includes thinking, so at high effort a small model can spend the whole reply thinking and end a turn with no action; this bounds that. |
| `claude_append_system_prompt` | Text appended to the CLI's system prompt (`--append-system-prompt`), e.g. an instruction to delegate to parallel subagents; the task prompt stays untouched. |
| `claude_extra_args` | Extra CLI arguments, appended verbatim before the query. |
| `session_header_name` | Header carrying the rollout session id; `null` disables. |
| `sandbox_provider`, `sandbox_config`, `sandbox_timeout` | Same contract as `opencode_sandboxed_agent`. |

The run verdict is carried on the verify response as `claude_status`
(`completed`, `error`, `timeout`, ...) with `claude_num_turns`,
`claude_is_error` and the raw CLI stdout/stderr, plus two delegation
counters: `claude_tool_use_counts` (calls per tool name, main thread and
Agent-tool subagents alike, so `Task`/`Agent` and `Workflow` entries say how
often the run delegated) and `claude_subagent_stats` (the result event's own
counter; it covers the Agent tool only, since Workflow-script agents do not
appear in the parent's stream).

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
