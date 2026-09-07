# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constraint-aware SWE-bench agent wrapper (multi-turn only).

Extends the OpenHands ``SWEBenchWrapper``: the full agentic rollout and
SWE-bench harness evaluation are inherited unchanged; on top, the completed
multi-turn trajectory is graded against if_format constraints and the
task reward is shaped::

    reward = task_reward * (1 + alpha * constraint_reward)

task_reward = 0 always yields reward 0 (no constraint reward hacking), and
constraint pressure never zeroes the task gradient on hard SWE tasks. When no
constraint had a gradeable step (any_graded=False), the constraint term is
dropped — "not measured" is not perfect compliance.

Canonical constraint ids, verifiers, and the reward formula live in the
sibling ``grading`` package — the single source of truth (the agentic-if
constraint-development repo re-exports it from here). The constraint
*instruction text* is expected to already be injected into the task prompt by
the dataset builder; this wrapper only grades and shapes.

Per-task declarations ride in ``responses_create_params.metadata`` (Responses
metadata values are strings, so structured values are JSON-encoded):

  constraints        JSON list: [{"type": "<canonical id>", "params": {...}}]
  constraint_alpha   optional float (default from config)
  grading_mode       optional: "fraction" (default) | "binary"
  step_aggregation   optional: "mean" (default) | "all"
  injection_mode     optional: "system_prompt" (default) | "first_user_turn"
                     | "mid_conversation"
  injection_step     optional int (with mid_conversation)
"""

from typing import Optional

from nemo_gym.base_resources_server import BaseMultiRewardVerifyResponse, BaseRunRequest
from responses_api_agents.swe_agents.app import (
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
)
from responses_api_agents.swe_agents_constrained.constrained_reward import (
    DEFAULT_CONSTRAINT_ALPHA,
    grade_and_shape,
)


class SWEBenchConstrainedWrapperConfig(SWEBenchWrapperConfig):
    constraint_alpha: float = DEFAULT_CONSTRAINT_ALPHA
    # "shaped" (default): reward = task * (1 + alpha * constraint_fraction)
    # "strict": reward = task * 1[all applicable constraints passed at every
    # applicable turn]; task failure and any violation both give 0 (Lin,
    # 2026-09-07 -- e2e GRPO recipe). Per-row metadata `reward_mode` overrides.
    reward_mode: str = "shaped"


class SWEBenchConstrainedVerifyResponse(SWEBenchVerifyResponse, BaseMultiRewardVerifyResponse):
    # Inherited from BaseMultiRewardVerifyResponse:
    #   reward: float
    #   reward_components: dict[str, float]
    task_reward: float = 0.0
    constraint_reward: Optional[float] = None
    # False when no constraint had a gradeable in-scope step: "format not
    # measured", NOT perfect compliance. The constraint term is dropped from
    # the reward in that case.
    constraint_graded: bool = False
    constraint_alpha: float = DEFAULT_CONSTRAINT_ALPHA
    # Per-constraint pass/fail, partial-credit scores, applicability, and
    # human-readable violations from grading.grade_constraints().
    constraint_results: dict[str, bool] = {}
    constraint_scores: dict[str, float] = {}
    constraint_applicable: dict[str, bool] = {}
    violations: list[str] = []
    # --- strict-mode / per-turn credit assignment (2026-09-07) ---
    reward_mode: str = "shaped"
    # True iff at least one constraint check ran and none failed.
    constraint_all_pass: bool = False
    # One entry per (constraint, graded step): {turn, step_index, constraint,
    # passed, kind, violation}. `turn` is the 1-based assistant turn, which
    # NeMo-RL maps onto its message_log (Gym-path logs alternate user/assistant,
    # so assistant message k is turn k+1) to push down exactly the violating
    # turns. Lists are not promoted to metrics; the ints below are.
    turn_verdicts: list[dict] = []
    first_violation_turn: Optional[int] = None
    num_graded_turns: int = 0
    num_violating_turns: int = 0


class SWEBenchConstrainedWrapper(SWEBenchWrapper):
    config: SWEBenchConstrainedWrapperConfig

    async def run(self, body: BaseRunRequest) -> SWEBenchConstrainedVerifyResponse:
        base = await super().run(body)
        metadata = dict(body.responses_create_params.metadata or {})
        fields = grade_and_shape(
            base.response.output,
            metadata,
            task_reward=base.reward,
            default_alpha=self.config.constraint_alpha,
            default_reward_mode=self.config.reward_mode,
        )
        return SWEBenchConstrainedVerifyResponse(**(base.model_dump() | fields))


if __name__ == "__main__":
    SWEBenchConstrainedWrapper.run_webserver()
