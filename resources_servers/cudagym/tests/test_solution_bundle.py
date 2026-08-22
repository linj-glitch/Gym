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

"""Unit tests for solution.json bundle normalization.

The tolerated bundle forms mirror what SolSwarm production accepts: the submit
skill's minimal ``{"sources": [...]}`` example, a ``{path: content}`` sources
map, a singular ``language`` key, and a bare ``entry_point`` function name.
"""

import pytest

from resources_servers.cudagym.solution_bundle import normalize_solution_bundle


META = {
    "language": "triton",
    "target_hardware": "B200",
    "destination_passing_style": False,
    "definition": {"name": "matmul_f16"},
    "workloads": [],
}


def test_sources_only_bundle_gets_full_envelope():
    """A bare sources-only bundle receives the full envelope from the row."""
    bundle = {"sources": [{"path": "run.py", "content": "def run(x, out): ..."}]}
    out = normalize_solution_bundle(bundle, META, "H100")
    assert out["name"] == "matmul_f16__submission_v1"
    assert out["definition"] == "matmul_f16"
    assert out["author"] == "agent"
    assert out["spec"]["languages"] == ["triton"]
    assert out["spec"]["target_hardware"] == ["B200"]  # row pin wins over default_sku
    assert out["spec"]["entry_point"] == "run.py::run"
    assert out["spec"]["destination_passing_style"] is False
    assert bundle == {"sources": [{"path": "run.py", "content": "def run(x, out): ..."}]}  # input untouched


def test_sources_map_is_converted_to_list():
    """A {path: content} sources map is converted to the list form."""
    out = normalize_solution_bundle({"sources": {"kernel.py": "def run(): pass"}}, META, "B200")
    assert out["sources"] == [{"path": "kernel.py", "content": "def run(): pass"}]
    assert out["spec"]["entry_point"] == "kernel.py::run"


def test_singular_language_and_bare_entry_point():
    """A singular `language` key and a bare entry-point name are normalized."""
    bundle = {
        "sources": [{"path": "impl.py", "content": "def go(): ..."}],
        "spec": {"language": "cute_dsl", "entry_point": "go"},
    }
    out = normalize_solution_bundle(bundle, META, "B200")
    # The row's language wins: an agent-declared language must not let a bundle
    # dodge the row's language requirement at reward time.
    assert out["spec"]["languages"] == [META["language"]]
    assert "language" not in out["spec"]
    assert out["spec"]["entry_point"] == "impl.py::go"


def test_full_bundle_passes_through_except_the_row_owned_spec_fields():
    """A complete bundle is preserved, except that the task row owns what the kernel is evaluated as."""
    bundle = {
        "name": "matmul_f16__agent_v3",
        "definition": "matmul_f16",
        "author": "agent_7",
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["H100"],
            "entry_point": "main.cpp::run",
            "destination_passing_style": True,
        },
        "sources": [{"path": "main.cpp", "content": "// ..."}],
    }
    out = normalize_solution_bundle(bundle, META, "B200")
    # Everything the agent set is preserved, except the evaluation-defining spec
    # fields, which the task row owns: languages (triton, not the bundle's
    # cuda_cpp), target_hardware (B200, not H100), and destination_passing_style
    # (False, not True).
    assert out == {
        **bundle,
        "spec": {
            **bundle["spec"],
            "languages": [META["language"]],
            "target_hardware": [META["target_hardware"]],
            "destination_passing_style": META["destination_passing_style"],
        },
    }


def test_agent_declared_hardware_never_wins():
    """A bundle declaring different hardware gets the row's value (else the server default).

    An agent-chosen SKU would defeat the endpoint SKU check by retargeting the
    build for other silicon.
    """
    bundle = {"sources": [{"path": "run.py", "content": "def run(): ..."}], "spec": {"target_hardware": "GB10"}}
    out = normalize_solution_bundle(bundle, META, "H100")
    assert out["spec"]["target_hardware"] == [META["target_hardware"]]  # row pin wins over the bundle
    out = normalize_solution_bundle(bundle, {"definition": {"name": "p"}, "destination_passing_style": False}, "H100")
    assert out["spec"]["target_hardware"] == ["H100"]  # no row pin -> default_sku, still not the bundle


def test_scalar_row_target_hardware_and_default_sku_fallback():
    """A scalar row target_hardware is wrapped; the default SKU applies when the row pins nothing."""
    out = normalize_solution_bundle(
        {"sources": [{"path": "run.py", "content": "def run(): ..."}]},
        {"definition": {"name": "p"}, "target_hardware": "GB10", "destination_passing_style": False},
        "H100",
    )
    assert out["spec"]["target_hardware"] == ["GB10"]  # scalar row value wrapped
    out = normalize_solution_bundle(
        {"sources": [{"path": "run.py", "content": "def run(): ..."}]},
        {"definition": {"name": "p"}, "destination_passing_style": False},
        "H100",
    )
    assert out["spec"]["target_hardware"] == ["H100"]  # no pin anywhere -> default_sku
    assert "languages" not in out["spec"]  # unknowable -> left for Solution validation to reject


def test_non_dict_passthrough_and_missing_definition_name():
    """Non-dict input passes through; a missing definition name gets a placeholder."""
    assert normalize_solution_bundle(["not", "a", "dict"], META, "B200") == ["not", "a", "dict"]
    out = normalize_solution_bundle(
        {"sources": [{"path": "run.py", "content": "x"}]}, {"destination_passing_style": True}, "B200"
    )
    assert out["name"] == "problem__submission_v1"
    assert out["definition"] == "problem"


def test_row_without_destination_passing_style_raises():
    """The row owns destination_passing_style; a row missing it must error, not default."""
    broken = {k: v for k, v in META.items() if k != "destination_passing_style"}
    with pytest.raises(KeyError):
        normalize_solution_bundle({"sources": [{"path": "run.py", "content": "x"}]}, broken, "B200")


def test_normalized_minimal_bundle_validates_as_solution():
    """The skill's minimal example must survive strict Solution validation post-normalization."""
    contracts = pytest.importorskip("cudagym.contracts.solution")
    bundle = {"sources": [{"path": "run.py", "content": "import torch\n\ndef run(x, out):\n    ...\n"}]}
    solution = contracts.Solution.model_validate(normalize_solution_bundle(bundle, META, "B200"))
    assert solution.spec.entry_point == "run.py::run"
    assert [h.value for h in solution.spec.target_hardware] == ["B200"]
