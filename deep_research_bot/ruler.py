'''
Copied from https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py
'''


from __future__ import annotations

from collections.abc import Iterable
from textwrap import dedent
from typing import Any

import art
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rich import print
import weave


class ToolCallPayload(BaseModel):
    """Lightweight container for tool call payloads."""

    model_config = ConfigDict(extra="allow")


class SerializableMessage(BaseModel):
    """Message representation with normalized tool calls."""

    role: str
    content: Any | None = None
    tool_calls: list[ToolCallPayload] | None = None

    model_config = ConfigDict(extra="allow")


class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""

    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(
        description="A short description of the trajectory's performance."
    )
    score: float = Field(description="A score between 0 and 1.")


class RulerResponse(BaseModel):
    """Response format expected from the LLM judge."""

    scores: list[TrajectoryScore] = Field(
        description="The scores for each trajectory."
    )


DEFAULT_RUBRIC = dedent(
    """\
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
)

_MESSAGE_LIST_ADAPTER = TypeAdapter(list[SerializableMessage])


def _normalize_tool_calls(tool_calls: Any | None) -> list[ToolCallPayload] | None:
    """Convert tool calls into JSON-serializable payloads."""
    if not tool_calls:
        return None

    normalized_calls: list[ToolCallPayload] = []
    candidates: Iterable[Any]
    if isinstance(tool_calls, list):
        candidates = tool_calls
    elif isinstance(tool_calls, Iterable):
        candidates = list(tool_calls)
    else:
        candidates = [tool_calls]

    for call in candidates:
        if isinstance(call, dict):
            payload: dict[str, Any] = call
        elif hasattr(call, "model_dump"):
            payload = call.model_dump(exclude_none=True)  # type: ignore[attr-defined]
        elif hasattr(call, "dict"):
            payload = call.dict(exclude_none=True)  # type: ignore[attr-defined]
        else:
            payload = dict(call)  # type: ignore[arg-type]
        normalized_calls.append(ToolCallPayload.model_validate(payload))
    return normalized_calls


def _sanitize_message(
    message: ChatCompletionMessageParam | dict[str, Any],
) -> SerializableMessage:
    """Return a sanitized copy of a chat message."""
    if hasattr(message, "model_dump"):
        raw = message.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    elif isinstance(message, dict):
        raw = message.copy()
    else:
        raw = dict(message)  # type: ignore[arg-type]

    raw["tool_calls"] = _normalize_tool_calls(raw.get("tool_calls"))
    return SerializableMessage.model_validate(raw)


def _sanitize_messages_for_serialization(
    messages: list[ChatCompletionMessageParam | dict[str, Any]],
) -> list[SerializableMessage]:
    return [_sanitize_message(message) for message in messages]


def _serialize_messages(messages: list[SerializableMessage]) -> str:
    # TypeAdapter.dump_json returns bytes; decode for downstream string joins.
    data = _MESSAGE_LIST_ADAPTER.dump_json(messages, exclude_none=True)
    return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)


def _extract_choice_content(choice: Any) -> str:
    """Handle both legacy string and content-block formats."""
    message = getattr(choice, "message", choice)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                maybe_text = getattr(block, "text", None)
                if isinstance(maybe_text, str):
                    text_parts.append(maybe_text)
        return "".join(text_parts)
    return ""

@weave.op
async def ruler(
    message_lists: list[list[ChatCompletionMessageParam | dict[str, Any]]],
    judge_model: str = "openai/o3",
    oai_client_kwargs: dict[str, Any] | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
    client: AsyncOpenAI | None = None,
) -> list[TrajectoryScore]:
    """Score a list of trajectories using an LLM judge.

    Args:
        message_lists: Trajectories represented as chat message sequences.
        judge_model: Model identifier for the judging call.
        oai_client_kwargs: Keyword arguments used to initialize `AsyncOpenAI`
            when a client instance is not explicitly provided.
        rubric: Rubric text guiding the judge.
        debug: When True, prints the raw JSON returned by the judge.
        client: Optional pre-configured AsyncOpenAI instance.
    """
    if not message_lists:
        return []

    sanitized_lists = [
        _sanitize_messages_for_serialization(messages) for messages in message_lists
    ]

    common_prefix_len = 0
    first_trajectory = sanitized_lists[0]
    for idx, template_message in enumerate(first_trajectory):
        if all(len(messages) > idx and messages[idx] == template_message for messages in sanitized_lists):
            common_prefix_len += 1
        else:
            break

    user_text_parts: list[str] = []
    if common_prefix_len > 0:
        common_prefix = first_trajectory[:common_prefix_len]
        user_text_parts.append("<context>")
        user_text_parts.append(_serialize_messages(common_prefix))
        user_text_parts.append("</context>\n")

    user_text_parts.append("Trajectories:\n")
    for idx, trajectory_messages in enumerate(sanitized_lists, start=1):
        trimmed = trajectory_messages[common_prefix_len:]
        serialized = _serialize_messages(trimmed)
        user_text_parts.append(f'<trajectory id="{idx}">\n{serialized}\n</trajectory>\n')

    user_text = "\n".join(user_text_parts).strip()

    judge_prompt = dedent(
        f"""
        All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
        """
    ).strip()

    request_messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    schema = RulerResponse.model_json_schema()
    schema["additionalProperties"] = False
    if "$defs" in schema:
        for definition in schema["$defs"].values():
            if isinstance(definition, dict):
                definition["additionalProperties"] = False

    response_client = client or AsyncOpenAI(**(oai_client_kwargs or {}))
    response = await response_client.chat.completions.create(
        model=judge_model,
        messages=request_messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RulerResponse",
                "strict": True,
                "schema": schema,
            },
        },
    )

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")
    first_choice = response.choices[0]

    raw_content = _extract_choice_content(first_choice)

    if debug:
        try:
            print("\n[RULER] Pretty-printed LLM choice JSON:")
            print(RulerResponse.model_validate_json(raw_content).model_dump())
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"[RULER] Could not parse choice content as JSON: {exc}")
            print(f"[RULER] Raw choice content: {raw_content}")

    parsed = RulerResponse.model_validate_json(raw_content)
    if len(parsed.scores) != len(sanitized_lists):
        raise ValueError(
            f"Received {len(parsed.scores)} scores for {len(sanitized_lists)} trajectories."
        )
    return list(parsed.scores)

@weave.op
async def ruler_score_group(
    group: art.TrajectoryGroup,
    judge_model: str = "openai/o4-mini",
    oai_client_kwargs: dict[str, Any] | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    swallow_exceptions: bool = False,
    debug: bool = False,
    client: AsyncOpenAI | None = None,
) -> art.TrajectoryGroup | None:
    """Score a trajectory group using the sanitized RULER implementation.

    Args:
        group: The trajectory batch to evaluate.
        judge_model: Model identifier for the judging call.
        oai_client_kwargs: Keyword arguments used to initialize `AsyncOpenAI`
            when a client instance is not explicitly provided.
        rubric: Rubric text guiding the judge.
        swallow_exceptions: When True, return `None` instead of raising on errors.
        debug: When True, prints the raw JSON returned by the judge.
        client: Optional pre-configured AsyncOpenAI instance.
    """
    sanitized_messages: list[list[dict[str, Any]]] = []
    new_trajectories: list[art.Trajectory] = []

    for trajectory in group.trajectories:
        sanitized_messages.append(
            [
                _sanitize_message(message).model_dump(by_alias=True, exclude_none=True)
                for message in trajectory.messages()
            ]
        )

        cloned = trajectory.__class__(
            messages_and_choices=trajectory.messages_and_choices,
            tools=trajectory.tools,
            additional_histories=[
                history.model_copy(deep=True)
                for history in trajectory.additional_histories
            ],
            reward=trajectory.reward,
            metrics=trajectory.metrics.copy(),
            metadata=trajectory.metadata.copy(),
            logs=trajectory.logs.copy(),
        )
        cloned.metrics["independent_reward"] = cloned.reward
        new_trajectories.append(cloned)

    try:
        scores = await ruler(
            sanitized_messages,
            judge_model=judge_model,
            oai_client_kwargs=oai_client_kwargs,
            rubric=rubric,
            debug=debug,
            client=client,
        )
    except Exception as exc:  # pragma: no cover - controlled by runtime flag
        if swallow_exceptions:
            print(f"[art_ruler] Swallowed exception: {exc}")
            return None
        raise

    for trajectory, score in zip(new_trajectories, scores):
        trajectory.metrics["ruler_score"] = score.score
        trajectory.reward = score.score
        trajectory.log(f"RULER explanation: {score.explanation}")

    return art.TrajectoryGroup(new_trajectories)


__all__ = [
    "DEFAULT_RUBRIC",
    "TrajectoryScore",
    "RulerResponse",
    "ruler",
    "ruler_score_group",
]

