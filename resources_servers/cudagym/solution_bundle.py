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

"""Normalization of agent-authored ``solution.json`` bundles.

A solution bundle is the JSON document an agent submits for evaluation. Its full
form is ``{name, definition, author, spec, sources}``, validated by
``cudagym.contracts.solution.Solution``. SolSwarm's platform accepts partial
bundles: the submit skill's minimal example is a bare ``{"sources": [...]}``, and
the platform supplies the ``name``, ``definition``, and ``spec`` envelope itself.
The cudagym CLI's ``python_solution_from_file`` builds the same envelope for bare
``.py`` files. This module gives the Gym cudagym resources server
(``resources_servers/cudagym/app.py``) the same tolerance, so an agent that
follows the skill's example is not rejected by strict ``Solution`` validation.

The module imports nothing beyond the standard library, so its unit tests run
without the cudagym or nemo_gym packages installed.
"""

from typing import Any


def normalize_solution_bundle(bundle: Any, meta: dict[str, Any], default_sku: str) -> Any:
    """Fill in the envelope of a partial solution.json from the rollout's problem.

    Existing keys are never overwritten, so a complete bundle passes through
    unchanged except for the evaluation-defining spec fields — ``languages``,
    ``target_hardware``, and ``destination_passing_style`` — which the task row
    always owns; an agent-declared value must not redefine what the kernel is
    evaluated as. Anything that is not a dict is returned as-is, so
    ``Solution.model_validate`` rejects it with its own error message.

    Args:
        bundle: The parsed agent-authored solution.json.
        meta: The rollout's ``verifier_metadata``. The ``definition``,
            ``language``, and ``target_hardware`` keys are consulted;
            ``destination_passing_style`` is required, and a row without it
            raises KeyError instead of guessing a value.
        default_sku: Fallback GPU SKU when the row does not pin
            ``target_hardware``.

    Returns:
        The normalized bundle, ready for ``Solution.model_validate``. The input
        dict is not mutated.
    """
    # Not a dict: pass through so Solution.model_validate produces its own error.
    if not isinstance(bundle, dict):
        return bundle
    bundle = dict(bundle)
    # Accept the {path: content} map form of sources; the list form passes through.
    sources = bundle.get("sources")
    if isinstance(sources, dict):
        bundle["sources"] = [{"path": p, "content": c} for p, c in sources.items()]
    # The first source's path completes a bare entry_point below.
    first_path = None
    if isinstance(bundle.get("sources"), list) and bundle["sources"]:
        first = bundle["sources"][0]
        if isinstance(first, dict):
            first_path = first.get("path")

    # Envelope defaults derived from the task row.
    definition_name = (meta.get("definition") or {}).get("name") or "problem"
    bundle.setdefault("name", f"{definition_name}__submission_v1")
    bundle.setdefault("definition", definition_name)
    bundle.setdefault("author", "agent")  # required by Solution; SolSwarm's envelope uses the file stem

    spec = bundle.get("spec")
    spec = dict(spec) if isinstance(spec, dict) else {}
    # Discard a singular 'language' key; the spec field is 'languages', filled below.
    spec.pop("language", None)
    # The task row's language is authoritative. An agent-declared language must
    # not override it: a bundle declaring "pytorch" for a triton row would
    # otherwise dodge the language requirement at reward time.
    languages = meta.get("language") or spec.get("languages")
    if isinstance(languages, str):
        languages = [languages]
    if languages:
        spec["languages"] = languages
    # Hardware: the row's, else the server default — never the bundle's own,
    # which would defeat the endpoint SKU check by retargeting the build.
    # Always a list.
    target_hardware = meta.get("target_hardware") or default_sku
    if isinstance(target_hardware, str):
        target_hardware = [target_hardware]
    spec["target_hardware"] = target_hardware
    # Complete a bare entry_point: 'func' becomes '<file>::func', and a missing one becomes '<file>::run'.
    entry_point = spec.get("entry_point")
    if isinstance(entry_point, str) and entry_point and "::" not in entry_point and first_path:
        spec["entry_point"] = f"{first_path}::{entry_point}"
    elif not entry_point and first_path:
        spec["entry_point"] = f"{first_path}::run"
    # Destination-passing style is likewise the row's choice, not the agent's:
    # it must match the Definition the server evaluates against. Hard-indexed,
    # like the final-file scoring path: a row without the key is malformed and
    # must fail the submission rather than default to a guess.
    spec["destination_passing_style"] = meta["destination_passing_style"]
    bundle["spec"] = spec
    return bundle
