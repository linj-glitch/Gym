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

"""Loopback Anthropic Messages proxy that turns document and image blocks into text.

Claude Code's Read tool attaches PDFs and images to the conversation as
``document`` / ``image`` content blocks. A text-only served model (or a proxy
translating to a text-only chat endpoint) rejects the whole request, and the
CLI ends the run on that API error: one ``Read /app/paper.pdf`` cost a full
Terminal-Bench rollout. The shim sits between the sandboxed CLI and the real
endpoint, rewrites those blocks into a short text notice that tells the model
to extract the content with command-line tools, and relays everything else
(streaming responses included) untouched.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from typing import Any

from aiohttp import web

from nemo_gym.server_utils import get_global_aiohttp_client


NON_TEXT_BLOCK_TYPES = ("document", "image")
# Hop-by-hop or recomputed headers that must not be copied between the two legs.
_DROP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


def _block_size_kib(block: dict[str, Any]) -> int | None:
    source = block.get("source") or {}
    data = source.get("data")
    if isinstance(data, str):
        try:
            return max(1, len(base64.b64decode(data, validate=False)) // 1024)
        except (ValueError, TypeError):
            return max(1, len(data) * 3 // 4 // 1024)
    return None


def placeholder_text(block: dict[str, Any]) -> str:
    source = block.get("source") or {}
    media_type = source.get("media_type") or block.get("type")
    size = _block_size_kib(block)
    size_text = f", {size} KB" if size is not None else ""
    return (
        f"[{block.get('type')} content omitted ({media_type}{size_text}): this model cannot view documents or "
        "images. Extract the content with command-line tools instead, e.g. pdftotext or a Python PDF/image "
        "library, and read the text output.]"
    )


def rewrite_content(content: Any) -> tuple[Any, int]:
    """Replace document/image blocks in a content value; returns (new_content, replaced_count)."""
    if not isinstance(content, list):
        return content, 0
    replaced = 0
    new_content = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in NON_TEXT_BLOCK_TYPES:
            new_content.append({"type": "text", "text": placeholder_text(block)})
            replaced += 1
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            inner, inner_replaced = rewrite_content(block.get("content"))
            replaced += inner_replaced
            new_content.append({**block, "content": inner} if inner_replaced else block)
        else:
            new_content.append(block)
    return new_content, replaced


def rewrite_messages_body(body: dict[str, Any], force_effort: str | None = None) -> tuple[dict[str, Any], int]:
    """Rewrite an Anthropic Messages request body; returns (body, number of changes made).

    Changes are document/image blocks replaced by text, plus one for the effort when
    ``force_effort`` is set and the request's ``output_config.effort`` differs. Claude Code
    sends its main-thread effort on most requests but "high" on some auxiliary and subagent
    calls; a served model whose template rejects "high" would otherwise fail exactly the
    delegated work an ultracode run is meant to measure.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, 0
    changed = 0
    new_messages = []
    for message in messages:
        if isinstance(message, dict):
            content, count = rewrite_content(message.get("content"))
            changed += count
            new_messages.append({**message, "content": content} if count else message)
        else:
            new_messages.append(message)
    new_body = {**body, "messages": new_messages} if changed else body
    if force_effort is not None:
        output_config = body.get("output_config") if isinstance(body.get("output_config"), dict) else {}
        if output_config.get("effort") != force_effort:
            new_body = {**new_body, "output_config": {**output_config, "effort": force_effort}}
            changed += 1
    return (new_body, changed) if changed else (body, 0)


class TextOnlyShim:
    """aiohttp server relaying every request to ``upstream_root`` after rewriting Messages bodies."""

    def __init__(
        self,
        upstream_root: str,
        port: int,
        host: str = "127.0.0.1",
        debug: bool = False,
        force_effort: str | None = None,
    ) -> None:
        self.upstream_root = upstream_root.rstrip("/")
        self.port = port
        self.host = host
        self.debug = debug
        self.force_effort = force_effort
        self._runner: web.AppRunner | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        async with self._lock:
            if self._runner is not None:
                return
            app = web.Application(client_max_size=512 * 1024 * 1024)
            app.router.add_route("*", "/{tail:.*}", self._relay)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            # Several agent workers may host one shim port; SO_REUSEPORT lets them share it.
            site = web.TCPSite(runner, self.host, self.port, reuse_port=True)
            await site.start()
            self._runner = runner
            print(f"Text-only shim listening on {self.base_url}, upstream {self.upstream_root}", file=sys.stderr)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _relay(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        if request.method == "POST" and body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                rewritten, changed = rewrite_messages_body(parsed, force_effort=self.force_effort)
                if changed:
                    body = json.dumps(rewritten).encode()
                    if self.debug:
                        print(f"Text-only shim made {changed} change(s) on {request.path}", file=sys.stderr)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}
        upstream_url = f"{self.upstream_root}{request.path_qs}"
        client = get_global_aiohttp_client()
        async with client.request(request.method, upstream_url, data=body, headers=headers, timeout=None) as upstream:
            response = web.StreamResponse(status=upstream.status)
            for key, value in upstream.headers.items():
                if key.lower() not in _DROP_HEADERS:
                    response.headers[key] = value
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
