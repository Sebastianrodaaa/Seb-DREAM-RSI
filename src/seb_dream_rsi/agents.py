"""Discovery agents.

The scripted agent calls the task's fixed proposer. The chat agent asks an
OpenAI-compatible endpoint for the next artifact JSON and still leaves scoring
to the task evaluator. Requests have a timeout and a short retry on transport
errors. The API key is read from the environment and is never written to a run.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from seb_dream_rsi.online import Proposal
from seb_dream_rsi.policy import PolicySpec


class ChatClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class ScriptedAgent:
    def __init__(self, task: Any) -> None:
        self.task = task

    def propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal:
        return self.task.scripted_propose(
            parent_artifact,
            branch_index=branch_index,
            depth=depth,
            trajectory=trajectory,
        )


def parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response JSON must be an object")
    return payload


class UrllibChatClient:
    """Minimal chat-completions client. No third-party HTTP library."""

    def __init__(
        self,
        *,
        model: str,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def complete(self, prompt: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                # Re-auth and bad requests will not succeed by waiting.
                if exc.code in {400, 401, 403, 404}:
                    raise RuntimeError(f"chat request failed with HTTP {exc.code}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"chat request failed: {last_error}")


class LLMAgent:
    """Ask a model for the next artifact. The evaluator still scores it."""

    def __init__(self, task: Any, client: ChatClient) -> None:
        self.task = task
        self.client = client

    def propose(
        self,
        parent_artifact: dict[str, Any],
        *,
        branch_index: int,
        depth: int,
        trajectory: list[dict[str, Any]],
    ) -> Proposal:
        prompt = (
            "You are the discovery agent inside a Dream-RSI loop. "
            "Propose the next candidate for one branch. Reply with a single JSON "
            "object {\"proposal\": string, \"artifact\": object}. Do not write "
            "the exploration policy.\n\n"
            f"Task: {self.task.name}\n"
            f"Branch: {branch_index}\n"
            f"Depth: {depth}\n"
            f"Parent artifact: {json.dumps(parent_artifact, sort_keys=True)}\n"
            f"Trajectory newest-last: {json.dumps(trajectory)}\n"
        )
        raw = self.client.complete(prompt)
        payload = parse_json_object(raw)
        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError("artifact must be a JSON object")
        proposal = str(payload.get("proposal") or "llm proposal")
        return Proposal(artifact=artifact, proposal=proposal)


def propose_policy_spec(client: ChatClient, incumbent: PolicySpec, feedback: str) -> PolicySpec:
    """Ask a model for the next PolicySpec. Invalid JSON keeps the caller in control."""

    prompt = (
        "Revise this Dream-RSI exploration policy. Reply with one JSON object "
        "and no other text. Allowed families: parallel_refine, adaptive. "
        "Fields: family, branch_count (1-64), refine_depth (1-64), "
        "patience (1-32), min_gain (0-10). Decisions at run time may use only "
        "the revealed prefix. Do not name a specific node id or target score.\n\n"
        f"Current spec: {json.dumps(incumbent.to_dict(), sort_keys=True)}\n"
        f"Replay feedback: {feedback}\n"
    )
    return PolicySpec.from_dict(parse_json_object(client.complete(prompt)))
