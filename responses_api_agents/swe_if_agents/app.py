# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""swe_if_agents: SWE-bench rollouts with injected instruction-following (IF) constraints, graded in the gym.

A thin wrapper around ``responses_api_agents.swe_agents`` (the OpenHands SWE agent). Every dataset row is a SWE
task plus, in its request metadata, the exact instruction surfaces the episode must run under and the constraints
to grade:

* ``tool_name_overrides``: the tool-name binding the episode exposes (exported to nv-OpenHands as
  ``TOOL_NAME_OVERRIDES``);
* ``system_prompt_template_text`` / ``user_prompt_template_text``: the prompts with the injected instruction,
  mounted per row;
* ``replay_observation_suffix``: an instruction appended to one recorded tool output of a replayed prefix
  (mid-task injection);
* ``sdg_item``: the constraints (verifier parameters), graded on the model-generated turns after the episode.

By default the task outcome reward is untouched: ``reward`` is the SWE-bench verdict (``reward_mode: outcome``);
a training recipe may fold the IF grades into it (shaped / strict / tiered, see ``if_constraints/reward.py``). The IF grades are attached as
``if_constraints`` (one record per constraint, per gradable step) for downstream aggregation. Grading semantics
live in ``if_constraints/`` (the constraint verifier package and its grader; see the README).

Requires an nv-OpenHands checkout that understands ``TOOL_NAME_OVERRIDES`` and ``observation_suffix`` (the fork
pinned in ``configs/swebench_opencode_if.yaml``).
"""

from typing import Any, Dict, List, Literal, Optional

import orjson
from pydantic import Field, model_validator

from responses_api_agents.swe_agents import app as swe
from responses_api_agents.swe_if_agents.hooks import (
    normalize_tool_name_overrides,
    tag_replay_observation_suffix,
    write_row_templates,
)
from responses_api_agents.swe_if_agents.if_constraints import grade_row
from responses_api_agents.swe_if_agents.if_constraints.reward import (
    DEFAULT_ALPHA,
    DEFAULT_PARTIAL,
    REWARD_MODES,
    compute_if_reward,
    resolve_reward_settings,
)

# The config's Literal has to spell the values out (a tuple cannot be spliced into Literal[...]); it is pinned to
# reward.REWARD_MODES here and in tests/test_app_config.py.
RewardMode = Literal["outcome", "shaped", "strict", "tiered", "gdpo", "gdpo_gated"]
assert tuple(RewardMode.__args__) == REWARD_MODES, "RewardMode drifted from reward.REWARD_MODES"


class SWEIFWrapperConfig(swe.SWEBenchWrapperConfig):
    if_grading: bool = Field(
        default=True,
        description=(
            "Grade the row's sdg_item constraints on the model-generated turns and attach if_constraints to the "
            "verify response."
        ),
    )
    empty_response_retries: int = Field(
        default=0,
        description=(
            "Exported to the agent as OPENCODE_EMPTY_RESPONSE_RETRIES: the OpenCode agent of the pinned nv-OpenHands fork "
            "re-issues the identical request up to this many times when the model returns neither content nor tool calls "
            "(a reasoning-only turn), instead of ending the episode on it. 0 (default) keeps the harness behaviour."
        ),
    )
    # ---- GRPO reward knob (2026-09-08). The benchmark leaves `reward` as the SWE verdict (outcome); a training
    # recipe selects how the if_constraints records fold into the scalar. Per-row metadata (reward_mode,
    # constraint_alpha, success_partial_reward; strings) overrides these. Semantics in if_constraints/reward.py.
    # Validated HERE, at server start: an episode costs 30-60 min, so a typo must not surface after the first rollout.
    reward_mode: RewardMode = Field(
        default="outcome",
        description=(
            "outcome: reward = SWE verdict, records attached only. shaped: task * (1 + alpha * mean step-average over "
            "applicable constraints). strict: task * 1[every applicable constraint all_pass] (nothing applicable -> 0). "
            "tiered: task * (1 if all pass else success_partial_reward); gdpo: task + constraint with reward_components for NeMo-RL GDPO. Any mode but outcome needs if_grading: true."
        ),
    )
    constraint_alpha: float = Field(default=DEFAULT_ALPHA, description="shaped mode multiplier on the constraint fraction")
    success_partial_reward: float = Field(
        default=DEFAULT_PARTIAL, ge=0.0, le=1.0, description="tiered mode credit for solved-but-violating, in [0, 1]"
    )

    @model_validator(mode="after")
    def _reward_mode_needs_grading(self) -> "SWEIFWrapperConfig":
        if self.reward_mode != "outcome" and not self.if_grading:
            raise ValueError(
                f"reward_mode={self.reward_mode!r} needs if_grading: true (the if_constraints records are the reward's input)"
            )
        return self


class SWEIFVerifyResponse(swe.SWEBenchVerifyResponse):
    # One record per constraint: {id, trigger, match, no_answer, instruction, n_steps, n_pass, n_silent, step_avg,
    # all_pass, graded_turns, continuation_only, steps: [{turn, reward, detail, items}]}. None when the row has no
    # constraints.
    if_constraints: Optional[List[Dict[str, Any]]] = None

    # ---- training scalars (if_constraints/reward.py). `reward` above is the outcome in outcome mode and the folded
    # value otherwise; these decompose it. Scalars are promoted to per-agent metrics by the trainer, lists are not.
    task_reward: float = 0.0
    constraint_reward: Optional[float] = None       # mean step-average over applicable constraints; None if none
    constraint_graded: bool = False                 # at least one constraint applicable ("not measured" is not compliance)
    constraint_all_pass: bool = False
    reward_mode: str = "outcome"
    constraint_alpha: float = DEFAULT_ALPHA
    success_partial_reward: float = 0.0
    n_constraints: int = 0                          # declared constraints (the grader's <grading_error> pseudo-record excluded)
    n_applicable: int = 0
    n_grading_errors: int = 0                       # records with `error`; > 0 => reward passed through as task (every mode)
    # True when this sample must not contribute to the gradient. NeMo-RL reads the mask from
    # full_result.instance_config.mask_sample (SWEBenchWrapperInstanceConfig, set by the base wrapper on agent/eval
    # timeouts and OOM; set here as well on a grading error in a non-outcome mode); this top-level copy mirrors that
    # final value only so it is promoted to a per-agent metric — nothing reads it for masking.
    mask_sample: bool = False
    # A per-row reward override (metadata reward_mode / constraint_alpha / success_partial_reward) that was rejected
    # and replaced by the config value; None when every override was accepted (see reward.resolve_reward_settings).
    reward_settings_error: Optional[str] = None
    continuation_only: bool = False
    # per graded step: {turn (1-based assistant turn in the frame of response.output), passed, constraint, kind, detail};
    # NeMo-RL maps `turn` onto its assistant message_log entries for grpo.constraint_violation_advantage.
    turn_verdicts: List[Dict[str, Any]] = Field(default_factory=list)
    first_violation_turn: Optional[int] = None
    num_graded_turns: int = 0
    num_violating_turns: int = 0
    reward_components: Dict[str, float] = Field(default_factory=dict)
    constraint_step_avgs: Dict[str, float] = Field(default_factory=dict)


class SWEIFWrapper(swe.SWEBenchWrapper):
    config: SWEIFWrapperConfig

    # ---- mid-task injection: tag the replayed tool message the instruction was appended to
    def _maybe_build_replay_messages(self, body: swe.NeMoGymResponseCreateParamsNonStreaming) -> Optional[str]:
        replay_json = super()._maybe_build_replay_messages(body)
        spec = (body.metadata or {}).get("replay_observation_suffix") if body.metadata else None
        if not spec:
            return replay_json
        if replay_json is None:
            raise ValueError("replay_observation_suffix was given but the request carries no trajectory to replay")
        messages = orjson.loads(replay_json)
        tag_replay_observation_suffix(messages, spec)
        return orjson.dumps(messages).decode()

    # ---- per-row instruction surfaces (tool binding, prompt templates) and the agent environment
    def _setup_params(self, body: swe.NeMoGymResponseCreateParamsNonStreaming):
        params, dataset_processor = super()._setup_params(body)
        md = body.metadata or {}
        changed = False
        agent_env: Dict[str, str] = {}
        overrides = normalize_tool_name_overrides(md.get("tool_name_overrides"))
        if overrides:
            agent_env["TOOL_NAME_OVERRIDES"] = overrides
        if self.config.empty_response_retries > 0:
            agent_env["OPENCODE_EMPTY_RESPONSE_RETRIES"] = str(self.config.empty_response_retries)
        if agent_env:
            params.resolved_agent_env = agent_env
            changed = True
        sp_path, up_path = write_row_templates(
            params.persistent_dir, md.get("system_prompt_template_text"), md.get("user_prompt_template_text")
        )
        if sp_path:
            params.resolved_system_prompt_template = sp_path
            changed = True
        if up_path:
            params.resolved_user_prompt_template = up_path
            changed = True
        if changed:
            if self.config.agent_framework != "openhands":
                raise ValueError(
                    "swe_if_agents needs agent_framework: openhands "
                    "(per-row tool names and templates are OpenHands features)"
                )
            # The base built the agent command before it knew about the row's surfaces and the agent environment:
            # rebuild from the amended params so the mounts and exports see them.
            params.agent_command = swe.OpenHandsHarnessProcessor(config=params).get_run_command()
            params.agent_apptainer_command_str = self._build_apptainer_command(params, params.agent_command)
            params.agent_script = params.agent_script_path.read_text()
        return params, dataset_processor

    # ---- grading: the IF grades ride along; `reward` is the outcome verdict in outcome mode and the fold otherwise
    async def run(self, body: swe.BaseRunRequest) -> SWEIFVerifyResponse:
        # Per-row overrides are resolved before the episode and never raise: the config values were validated at
        # server start (SWEIFWrapperConfig), and a malformed row value falls back to them with the error recorded on
        # the response — a 30-60 min rollout is not lost to a bad metadata string.
        mode, alpha, partial, settings_error = resolve_reward_settings(
            body.responses_create_params.metadata,
            self.config.reward_mode,
            self.config.constraint_alpha,
            self.config.success_partial_reward,
            if_grading=self.config.if_grading,
        )
        base = await super().run(body)
        records = None
        if self.config.if_grading:
            has_output = base.response is not None and base.response.output
            output_items = [o.model_dump() for o in base.response.output] if has_output else []
            # responses_create_params is validated into a model by the verify response; accept a dict too
            rcp = base.responses_create_params
            raw_input = rcp.get("input") if isinstance(rcp, dict) else getattr(rcp, "input", None)
            input_items = [i.model_dump() if hasattr(i, "model_dump") else i for i in (raw_input or [])]
            records = grade_row(body.responses_create_params.metadata or {}, input_items, output_items)
        folded = compute_if_reward(records, base.reward, mode, alpha, partial)
        fields = base.model_dump()
        fields["reward"] = folded.pop("reward")
        # A grading error in a training mode: the reward passed through as the task verdict (see reward.py) and the
        # sample is masked from the gradient through the base wrapper's flag, the one NeMo-RL reads
        # (full_result.instance_config.mask_sample). The top-level `mask_sample` mirrors the final flag as a metric.
        grading_mask = bool(folded.pop("mask_sample"))
        ic = fields.get("instance_config")
        if not isinstance(ic, dict):
            ic = ic.model_dump() if hasattr(ic, "model_dump") else dict(ic or {})
        ic["mask_sample"] = bool(ic.get("mask_sample", False)) or grading_mask
        fields["instance_config"] = ic
        return SWEIFVerifyResponse(
            **fields,
            if_constraints=records,
            **folded,
            mask_sample=ic["mask_sample"],
            reward_settings_error=settings_error,
        )


if __name__ == "__main__":
    SWEIFWrapper.run_webserver()
