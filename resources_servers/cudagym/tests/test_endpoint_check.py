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

"""Unit tests for endpoint selection and the endpoint SKU check.

``endpoints`` maps each GPU the server scores on to that GPU's evaluation
endpoint, so a task row is always evaluated on the silicon its anchors were
measured on. These tests cover which URL a SKU resolves to, that the SDK client
is built once per endpoint, and how the server reacts to the /health check's
three verdicts (config ``verify_endpoint_sku``), including that it checks every
server a pooled /health lists. The comparison itself is
``cudagym.rl.verify_health_payload`` and is tested there.
"""

import asyncio
import types

import pytest


pytest.importorskip("nemo_gym")
pytest.importorskip("cudagym")  # the shared rl helpers the app module imports at load time

from resources_servers.cudagym import app as app_module  # noqa: E402
from resources_servers.cudagym.app import CudaGymResourcesServer  # noqa: E402


class _StubClient:
    """CudaGym SDK client stand-in that returns a canned /health response."""

    def __init__(self, resp):
        self._resp = resp

    async def health(self):
        return self._resp


def _run_check(resp, sku="B200"):
    """Run the endpoint SKU check against a canned /health response."""
    return asyncio.run(CudaGymResourcesServer._verify_endpoint_sku(types.SimpleNamespace(), _StubClient(resp), sku))


def test_check_passes_on_matching_unified_payload():
    """A single unified /health payload with the declared GPU passes."""
    _run_check({"status": "healthy", "health": {"gpu_model": "NVIDIA B200", "sm_version": "sm_100"}})


def test_check_tolerates_compile_only_responders_in_server_list():
    """Every listed server is checked; one with no GPU fields warns but does not fail."""
    _run_check(
        {
            "status": "healthy",
            "servers": [
                {"health": {"gpu_model": None, "sm_version": None}},  # compile-only responder
                {"health": {"gpu_model": "NVIDIA B200", "sm_version": "sm_100"}},
            ],
        }
    )


def test_check_tolerates_empty_server_list():
    """Regression: an empty servers list warns instead of raising NameError."""
    _run_check({"status": "healthy", "servers": []})


def test_check_raises_on_mismatch_and_unhealthy():
    """A wrong GPU model or an unhealthy status raises RuntimeError."""
    with pytest.raises(RuntimeError, match="reports different silicon"):
        _run_check({"status": "healthy", "health": {"gpu_model": "NVIDIA H100"}})
    with pytest.raises(RuntimeError, match="unhealthy"):
        _run_check({"status": "unhealthy", "health": {}})


# --- endpoint selection: one URL and one client per SKU --


class _EndpointServer:
    """Server stand-in running the real endpoint resolution and client cache."""

    _endpoint_url = CudaGymResourcesServer._endpoint_url
    _client_for = CudaGymResourcesServer._client_for
    _verify_endpoint_sku = CudaGymResourcesServer._verify_endpoint_sku

    def __init__(self, endpoints):
        self.config = types.SimpleNamespace(endpoints=dict(endpoints), verify_endpoint_sku=True)
        self._clients = {}
        self._clients_lock = asyncio.Lock()
        self._endpoint_sku_checked = set()


class _RecordingClient:
    """SDK client stand-in that records its URLs and counts /health calls."""

    def __init__(self, *, compile_server_url, gpu_server_url):
        self.compile_server_url = compile_server_url
        self.gpu_server_url = gpu_server_url
        self.health_calls = 0

    async def health(self):
        self.health_calls += 1
        # No GPU fields: the check reports "unverifiable" and warns, which keeps
        # this test about routing rather than about the health verdicts above.
        return {"status": "healthy", "health": {}}

    async def close(self):
        self.closed = True
        return None


def test_concurrent_rollouts_share_one_live_client_per_sku(monkeypatch):
    """Evaluations must not close the shared client, and must not race to build it.

    The client is cached for the life of the server, so a caller that closed it
    would kill the session other rollouts on that GPU are using mid-request; and
    two rollouts that miss the cache together must not each build one, which
    would leak whichever session lost the write.
    """
    monkeypatch.setattr(app_module, "Client", _RecordingClient)
    server = _EndpointServer({"B200": "http://b200.test"})

    async def scenario():
        clients = await asyncio.gather(*(server._client_for("B200") for _ in range(8)))
        # One client, built once, health-checked once, and still open.
        assert len({id(c) for c in clients}) == 1
        assert clients[0].health_calls == 1
        assert not getattr(clients[0], "closed", False)
        return clients[0]

    client = asyncio.run(scenario())
    # A later evaluation gets the same live client rather than a closed one.
    assert asyncio.run(server._client_for("B200")) is client
    assert not getattr(client, "closed", False)


def test_each_sku_gets_its_own_endpoint(monkeypatch):
    """A row's GPU decides its endpoint, and each endpoint is built and checked once."""
    monkeypatch.setattr(app_module, "Client", _RecordingClient)
    server = _EndpointServer({"B200": "http://b200.test", "H100": "http://h100.test"})

    b200 = asyncio.run(server._client_for("B200"))
    h100 = asyncio.run(server._client_for("H100"))
    assert (b200.compile_server_url, b200.gpu_server_url) == ("http://b200.test", "http://b200.test")
    assert (h100.compile_server_url, h100.gpu_server_url) == ("http://h100.test", "http://h100.test")
    # One HTTP session per endpoint for the life of the server, not one per
    # submission, and the /health check costs one request per endpoint.
    assert asyncio.run(server._client_for("B200")) is b200
    assert b200.health_calls == 1 and h100.health_calls == 1


def test_empty_endpoint_url_resolves_from_the_environment(monkeypatch):
    """An empty URL is an address that only exists once the job is running."""
    server = _EndpointServer({"B200": ""})
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://allocated.test")
    assert server._endpoint_url("B200") == "http://allocated.test"
    # CUDAGYM_URL is SolSwarm production's name for the same address.
    monkeypatch.delenv("CUDAGYM_UNIFIED_SERVER_URL")
    monkeypatch.setenv("CUDAGYM_URL", "http://legacy.test")
    assert server._endpoint_url("B200") == "http://legacy.test"
    # Neither configured nor in the environment: refuse instead of evaluating
    # against whatever default the SDK would pick.
    monkeypatch.delenv("CUDAGYM_URL")
    with pytest.raises(RuntimeError, match="B200"):
        server._endpoint_url("B200")


def test_configured_url_wins_over_the_environment(monkeypatch):
    """A configured endpoint is used even when the fallback variable is set."""
    monkeypatch.setenv("CUDAGYM_UNIFIED_SERVER_URL", "http://allocated.test")
    server = _EndpointServer({"B200": "http://b200.test", "H100": ""})
    assert server._endpoint_url("B200") == "http://b200.test"
    assert server._endpoint_url("H100") == "http://allocated.test"


def test_sdk_client_accepts_our_construction_kwargs():
    """The server builds Client(compile_server_url=..., gpu_server_url=...) — no auth kwarg.

    The 2.4.3 SDK resolves credentials from the environment per target; a
    constructor-signature change here broke every submission evaluation once
    (auth_token=), so the exact call shape is pinned against the real class.
    """
    import inspect

    from cudagym.sdk.client import Client

    params = inspect.signature(Client.__init__).parameters
    assert "compile_server_url" in params and "gpu_server_url" in params
    assert "auth_token" not in params
