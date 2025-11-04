# Global Configuration & Setup
import inspect
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import partial
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError
from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel

import art
class Console(RichConsole):
    def md(self, text): 
        return self.print(Markdown(text))

console = Console()

def estimate_token_count(messages: list[dict[str, Any]]) -> int:
    """
    Estimate token count for messages using character-based heuristic. 4 tokens per character.
    """
    total_chars = 0
    
    for message in messages:
        # Convert entire message to string and count characters
        # This includes role, content, and any other fields
        message_str = json.dumps(message)
        total_chars += len(message_str)
    
    # Rough heuristic: 4 characters ≈ 1 token
    base_estimate = total_chars / 4
    
    # Add 10% overhead for message formatting 
    # (things like <|start|>assistant, etc.)
    with_overhead = base_estimate * 1.1
    
    return int(with_overhead)


def _unwrap_callable(func: Callable) -> Callable:
    """Return the underlying function for wrappers and partials."""
    base_func = inspect.unwrap(func)

    # inspect.unwrap stops at functools.partial, so peel them manually
    while isinstance(base_func, partial):
        base_func = inspect.unwrap(base_func.func)

    return base_func


def _callable_name(func: Callable) -> str:
    """Best-effort friendly name for a callable, even if wrapped."""
    base_func = _unwrap_callable(func)
    return getattr(base_func, "__name__", base_func.__class__.__name__)


def _generate_tool_schema(func: Callable) -> dict:
    """Given a Python function, generate a tool-compatible JSON schema.
    Handles basic types and Enums. Assumes docstrings are formatted for arg descriptions.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    base_func = _unwrap_callable(func)
    type_hints = get_type_hints(base_func)

    schema = {
        "type": "function",
        "function": {
            "name": _callable_name(func),
            "description": inspect.getdoc(base_func).split("\\n")[0] if inspect.getdoc(base_func) else "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    docstring = inspect.getdoc(base_func)
    param_descriptions = {}
    if docstring:
        args_section = False
        current_param = None
        for line in docstring.split('\\n'):
            line_stripped = line.strip()
            if line_stripped.lower().startswith(("args:", "arguments:", "parameters:")):
                args_section = True
                continue
            if args_section:
                if ":" in line_stripped:
                    param_name, desc = line_stripped.split(":", 1)
                    param_descriptions[param_name.strip()] = desc.strip()
                elif line_stripped and not line_stripped.startswith(" "): # Heuristic: end of args section
                     args_section = False

    for name, param in parameters.items():
        is_required = param.default == inspect.Parameter.empty
        param_type = type_hints.get(name, Any)
        json_type = "string"
        param_schema = {}

        # Basic type mapping
        if param_type == str: json_type = "string"
        elif param_type == int: json_type = "integer"
        elif param_type == float: json_type = "number"
        elif param_type == bool: json_type = "boolean"
        elif hasattr(param_type, '__origin__') and param_type.__origin__ is list: # Handle list[type]
             item_type = param_type.__args__[0] if param_type.__args__ else Any
             if item_type == str: param_schema = {"type": "array", "items": {"type": "string"}}
             elif item_type == int: param_schema = {"type": "array", "items": {"type": "integer"}}
             # Add more list item types if needed
             else: param_schema = {"type": "array", "items": {"type": "string"}} # Default list item type
        elif hasattr(param_type, "__members__") and issubclass(param_type, Enum): # Handle Enum
             json_type = "string"
             param_schema["enum"] = [e.value for e in param_type]

        if not param_schema: # If not set by list or Enum
            param_schema["type"] = json_type

        param_schema["description"] = param_descriptions.get(name, "")

        if param.default != inspect.Parameter.empty and param.default is not None:
             param_schema["default"] = param.default # Note: OpenAI schema doesn't officially use default, but useful metadata

        schema["function"]["parameters"]["properties"][name] = param_schema
        if is_required:
            schema["function"]["parameters"]["required"].append(name)
    return schema

def _get_tool(tools: list[Callable], name: str) -> Callable:
    for t in tools:
        if t.__name__ == name:
            return t
    raise KeyError(f"No tool with name {name} found")


def function_tool(func: Callable) -> Callable:
    """Attaches a tool schema to the function and marks it as a tool.
    Call this *after* defining your function: my_func = function_tool(my_func)
    """
    base_name = _callable_name(func)
    try:
        func.tool_schema = _generate_tool_schema(func)
        func.is_tool = True # Mark it as a tool

        # Ensure wrapped callables expose a __name__ attribute for lookup
        if not hasattr(func, "__name__") or func.__name__ == func.__class__.__name__:
            try:
                func.__name__ = base_name
            except AttributeError:
                pass
    except Exception as e:
        console.print(f"Error processing tool {base_name}: {e}")
        # Optionally raise or mark as failed
        func.tool_schema = None
        func.is_tool = False
    return func

def perform_tool_calls(tools: list[Callable], tool_calls: list[Any]) -> list[dict]:
    "Perform the tool calls and return the messages with the tool call results"
    messages = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Invalid JSON in tool call arguments for {function_name}[/red]")
            console.print(f"[dim]Error: {e}[/dim]")
            console.print(f"[dim]Arguments: {tool_call.function.arguments[:200]}...[/dim]")
            
            # Return a simple, safe error message to the agent so it can recover
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": f"Error: The JSON format for {function_name} was invalid. Please ensure the argument is properly formatted JSON with all text inside the parameter value.",
            })
            # no reason to continue if the JSON is invalid
            return messages
        
        try:
            with console.status(f"[bold cyan]Executing {function_name}...[/bold cyan]"):
                tool = _get_tool(tools, function_name)
                tool_response = tool(**function_args)
            
            # Create panel content
            panel_content = f"[bold cyan]🔧 Tool Call:[/bold cyan] {function_name}\n\n"
            panel_content += f"[dim]Args: {tool_call.function.arguments}[/dim]\n\n"
            
            if isinstance(tool_response, list):
                panel_content += f"[green]✓[/green] Found {len(tool_response)} results"
            else:
                panel_content += f"[green]✓[/green] {function_name} executed successfully"
            
            console.print(Panel(panel_content, border_style="cyan"))
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": str(tool_response),
            })
        except Exception as e:
            console.print(f"[red]✗ Error executing {function_name}: {e}[/red]")
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": f"Error executing tool: {str(e)}",
            })
            
    return messages

def _to_plain(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_to_plain(v) for v in obj]
    return obj


LOGPROB_KEYS: set[str] = {"logprobs", "top_logprobs"}


def _strip_logprob_fields(value: Any) -> Any:
    """Remove logprob-related keys from nested message structures."""
    if isinstance(value, dict):
        return {
            key: _strip_logprob_fields(nested)
            for key, nested in value.items()
            if key not in LOGPROB_KEYS
        }

    if isinstance(value, list):
        return [_strip_logprob_fields(item) for item in value]

    if isinstance(value, BaseModel):
        sanitized = _strip_logprob_fields(value.model_dump())
        try:
            return value.__class__.model_validate(sanitized)
        except ValidationError:
            return sanitized

    return value


def _sanitize_messages_and_choices(
    messages_and_choices: Sequence[Any],
) -> list[Any]:
    return [_strip_logprob_fields(message) for message in messages_and_choices]


def _prepare_group_for_judge(group: art.TrajectoryGroup) -> art.TrajectoryGroup:
    sanitized_trajectories: list[art.Trajectory] = []
    for trajectory in group.trajectories:
        sanitized_histories = [
            history.__class__(
                messages_and_choices=_sanitize_messages_and_choices(
                    history.messages_and_choices
                ),
                tools=history.tools,
            )
            for history in trajectory.additional_histories
        ]

        sanitized_trajectories.append(
            art.Trajectory(
                messages_and_choices=_sanitize_messages_and_choices(
                    trajectory.messages_and_choices
                ),
                tools=list(trajectory.tools) if trajectory.tools else None,
                additional_histories=sanitized_histories,
                reward=trajectory.reward,
                metrics=trajectory.metrics.copy(),
                auto_metrics=trajectory.auto_metrics.copy(),
                metadata=trajectory.metadata.copy(),
                logs=trajectory.logs.copy(),
            )
        )

    judge_ready_group = art.TrajectoryGroup(sanitized_trajectories)
    judge_ready_group.exceptions = group.exceptions  # type: ignore[attr-defined]
    return judge_ready_group


def _restore_logprobs(
    original_group: art.TrajectoryGroup,
    judged_group: art.TrajectoryGroup,
) -> art.TrajectoryGroup:
    if len(original_group.trajectories) != len(judged_group.trajectories):
        raise ValueError(
            "Judge returned a different number of trajectories than provided"
        )

    updated_trajectories: list[art.Trajectory] = []
    for original, judged in zip(
        original_group.trajectories, judged_group.trajectories
    ):
        updated_trajectories.append(
            art.Trajectory(
                messages_and_choices=list(original.messages_and_choices),
                tools=list(original.tools) if original.tools else None,
                additional_histories=[
                    history.model_copy(deep=True)
                    for history in original.additional_histories
                ],
                reward=judged.reward,
                metrics=judged.metrics.copy(),
                auto_metrics=judged.auto_metrics.copy(),
                metadata=original.metadata.copy(),
                logs=judged.logs.copy(),
            )
        )

    restored_group = art.TrajectoryGroup(updated_trajectories)
    restored_group.exceptions = original_group.exceptions  # type: ignore[attr-defined]
    return restored_group