# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import socket

from aiohttp import ClientSession, web

import nemo_gym.server_utils as server_utils
from responses_api_agents.claude_code_sandboxed_agent.text_only_shim import (
    TextOnlyShim,
    placeholder_text,
    rewrite_messages_body,
)


PDF_BLOCK = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(b"%PDF" * 600).decode()},
}


def test_rewrite_replaces_document_and_image_blocks_everywhere() -> None:
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "plain string stays"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "PDF read"}, PDF_BLOCK],
                    },
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
                    {"type": "text", "text": "keep"},
                ],
            },
        ],
    }
    rewritten, replaced = rewrite_messages_body(body)

    assert replaced == 2
    blocks = rewritten["messages"][1]["content"]
    assert blocks[0]["content"][1]["type"] == "text"
    assert "application/pdf" in blocks[0]["content"][1]["text"]
    assert "2 KB" in blocks[0]["content"][1]["text"]
    assert blocks[1] == {"type": "text", "text": placeholder_text(body["messages"][1]["content"][1])}
    assert blocks[2] == {"type": "text", "text": "keep"}
    assert rewritten["messages"][0] == body["messages"][0]
    # The original is untouched.
    assert body["messages"][1]["content"][1]["type"] == "image"


def test_rewrite_leaves_text_only_bodies_alone() -> None:
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "max_tokens": 5}
    rewritten, replaced = rewrite_messages_body(body)
    assert replaced == 0 and rewritten is body
    assert rewrite_messages_body({"no": "messages"}) == ({"no": "messages"}, 0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_shim_rewrites_and_relays_streaming_responses(monkeypatch) -> None:
    seen: list[dict] = []

    async def upstream_handler(request: web.Request) -> web.StreamResponse:
        seen.append(json.loads(await request.read()))
        assert request.headers["x-switchyard-session-id"] == "sess-1"
        response = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for event in (b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n"):
            await response.write(event)
        await response.write_eof()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/messages", upstream_handler)
    upstream_runner = web.AppRunner(upstream_app)
    await upstream_runner.setup()
    upstream_port = _free_port()
    await web.TCPSite(upstream_runner, "127.0.0.1", upstream_port).start()

    async with ClientSession() as session:
        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
        import responses_api_agents.claude_code_sandboxed_agent.text_only_shim as shim_module

        monkeypatch.setattr(shim_module, "get_global_aiohttp_client", lambda: session)
        shim = TextOnlyShim(upstream_root=f"http://127.0.0.1:{upstream_port}", port=_free_port())
        await shim.start()
        try:
            body = {
                "model": "m",
                "messages": [{"role": "user", "content": [PDF_BLOCK, {"type": "text", "text": "q"}]}],
            }
            async with session.post(
                f"{shim.base_url}/v1/messages?beta=true",
                json=body,
                headers={"x-switchyard-session-id": "sess-1"},
            ) as response:
                assert response.status == 200
                assert response.headers["content-type"] == "text/event-stream"
                relayed = await response.read()
        finally:
            await shim.stop()
            await upstream_runner.cleanup()

    assert relayed == b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"
    assert len(seen) == 1
    assert seen[0]["messages"][0]["content"][0]["type"] == "text"
    assert "application/pdf" in seen[0]["messages"][0]["content"][0]["text"]
    assert seen[0]["messages"][0]["content"][1] == {"type": "text", "text": "q"}


def test_force_effort_rewrites_only_when_different() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "high", "x": 1}}
    rewritten, changed = rewrite_messages_body(body, force_effort="xhigh")
    assert changed == 1 and rewritten["output_config"] == {"effort": "xhigh", "x": 1}
    assert body["output_config"]["effort"] == "high"
    same, changed = rewrite_messages_body(rewritten, force_effort="xhigh")
    assert changed == 0 and same is rewritten
    no_config = {"messages": [{"role": "user", "content": "hi"}]}
    forced, changed = rewrite_messages_body(no_config, force_effort="medium")
    assert changed == 1 and forced["output_config"] == {"effort": "medium"}
