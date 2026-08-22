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

"""Unit test for the bytes-safe request-validation handler.

A request whose body cannot be parsed reaches the RequestValidationError
handler with the raw ``bytes`` body inside ``exc.errors()[i]["input"]`` and
``exc.body``. The base server's handler json.dumps()es those and crashes (and
FastAPI's default 422 encoder crashes on non-UTF8 bytes), converting the 422
into a 500 that hides which field failed; ``log_validation_error`` logs
bytes-safely and returns the standard 422 itself.
"""

import asyncio
import json
import types

import pytest


pytest.importorskip("nemo_gym")
pytest.importorskip("cudagym")

from fastapi.exceptions import RequestValidationError  # noqa: E402

from resources_servers.cudagym.app import log_validation_error  # noqa: E402


def _bytes_validation_error():
    """A RequestValidationError as produced for an unparseable (bytes) body."""
    return RequestValidationError(
        [{"type": "model_attributes_type", "loc": ("body",), "msg": "Input is not a valid dict", "input": b"\x80raw"}],
        body=b"\x80raw",
    )


def test_handler_answers_422_where_plain_dumps_crashes(caplog):
    """The exact input that breaks json.dumps still yields a logged 422."""
    exc = _bytes_validation_error()
    # The premise: the base handler's serialization crashes on this error.
    with pytest.raises(TypeError):
        json.dumps(exc.errors())

    request = types.SimpleNamespace(method="POST", url=types.SimpleNamespace(path="/rollout/t/api/v1/evaluate"))
    with caplog.at_level("ERROR"):
        response = asyncio.run(log_validation_error(request, exc))

    assert response.status_code == 422
    assert json.loads(response.body)["detail"][0]["msg"] == "Input is not a valid dict"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "/rollout/t/api/v1/evaluate" in logged
    assert "not a valid dict" in logged
