# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the minimal-profile prompt contract (``CudaAgent._build_prompt``).

``_build_prompt`` only reads ``self.config``, so it is exercised unbound with a
duck-typed stub — no server/config bootstrap needed.
"""

import types

import pytest


pytest.importorskip("nemo_gym")
pytest.importorskip("cudagym")  # the shared rl helpers the app module imports at load time

from responses_api_agents.cuda_agent.app import CudaAgent  # noqa: E402


META = {"language": "triton", "target_hardware": "B200"}


def _stub(system_prompt=None, submission_mode="final_file", sandbox_instructions=None):
    config = types.SimpleNamespace(
        system_prompt=system_prompt,
        submission_mode=submission_mode,
        sandbox_instructions=sandbox_instructions,
        # Empty: the row's own target_hardware then names the GPU, the
        # single-endpoint arrangement these tests use.
        cudagym_endpoints={},
    )
    agent = types.SimpleNamespace(config=config)
    agent._row_sku = lambda meta: CudaAgent._row_sku(agent, meta)
    return agent


def test_contract_states_sku_and_language_and_gpu_flag():
    prompt = CudaAgent._build_prompt(_stub(), "Optimize X.", None, META)
    assert prompt.startswith("Optimize X.\n\n# Sandbox contract")
    assert "- Target GPU: B200. Kernel language: triton.\n" in prompt
    assert "--solution kernel.py" in prompt
    assert "--gpu B200" in prompt


def test_cpp_language_gets_solution_json_command():
    prompt = CudaAgent._build_prompt(_stub(), "u", None, {"language": "cuda_cpp", "target_hardware": "H100"})
    assert "- Target GPU: H100. Kernel language: cuda_cpp.\n" in prompt
    assert "--solution solution.json" in prompt
    # .py auto-wrap (and its --gpu flag) does not apply to C++-family kernels.
    assert "--gpu" not in prompt
    assert "languages: ['cuda_cpp']" in prompt
    assert "main.cpp::run" in prompt


def test_config_system_prompt_folds_in_like_base_agent():
    prompt = CudaAgent._build_prompt(_stub(system_prompt="SYS."), "user msg", None, META)
    assert prompt.startswith("SYS.\n\nuser msg")
    both = CudaAgent._build_prompt(_stub(system_prompt="SYS."), "user msg", "ROW-SYS.", META)
    assert both.startswith("SYS.\n\nROW-SYS.\n\nuser msg")
    neither = CudaAgent._build_prompt(_stub(), "user msg", None, META)
    assert neither.startswith("user msg")


def test_unknown_language_raises_instead_of_staging_the_wrong_file():
    """A language ``cudagym.rl`` does not know can only come from a malformed task row.

    Defaulting to kernel.py would stage a filename the resources server does not
    read for that row, and the resulting 0 would be recorded as the model's
    failure, so the prompt builder lets the lookup raise and the rollout fails at
    its start instead.
    """
    with pytest.raises(ValueError, match="'rust'.*LANGUAGE_DEFAULTS"):
        CudaAgent._build_prompt(_stub(), "u", None, {"language": "rust", "target_hardware": "H100"})


def test_solswarm_submit_contract_teaches_plural_spec_fields():
    prompt = CudaAgent._build_prompt(_stub(submission_mode="solswarm_submit"), "u", None, META)
    assert "languages: [...]" in prompt
    assert "target_hardware: [...]" in prompt
    assert "sources-only bundle also works" in prompt
    assert "submit.py --solution solution.json" in prompt
